// 块编辑器（可移植内核：渲染、contenteditable 编辑、块操作、流式更新）。
// 单一数据源 = 块数组；块边界是 DOM 元素而非文本标记——不存在 offset
// 对账、标记误删问题。流式追加直接改块内容后重渲染该块。
import {
  TYPE_LABEL,
  NO_PPL_COLOR,
  pplColor,
  reconcilePpl,
  normalizeCotBlock,
  newBlock,
  ensureActive,
} from "./model.js?v=2";

const BLOCK_ACTIONS = [
  { type: "prompt", label: "指令", hint: "用户输入，→ user 消息" },
  { type: "generate", label: "正文", hint: "模型生成，→ assistant 消息" },
  { type: "system", label: "系统", hint: "系统提示词（技能固化）" },
  { type: "cot", label: "思考", hint: "思维链（默认不进上下文）" },
];

export class BlockEditor {
  /**
   * @param root 块流容器元素
   * @param opts { onChange(): 文档变更（脏标记/自动保存） }
   */
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = opts;
    this.blocks = [];
    this.editable = true; // 生成中关闭编辑
    this._menu = null;
    this.render();
  }

  // ---------------------------------------------------------------- 数据

  setBlocks(blocks) {
    this.blocks = ensureActive(blocks || []);
    for (const b of this.blocks) {
      if (!b.id) b.id = "b" + Math.random().toString(36).slice(2, 10);
      // 折叠仅为 UI 态：加载的块未显式指定时，系统/思考默认折叠
      if (b.collapsed === undefined) {
        b.collapsed = b.type === "cot" || b.type === "system";
      }
      // 旧格式 cot（标签写在内容里）归一化为 纯正文 + closed 属性
      normalizeCotBlock(b);
    }
    this.render();
  }

  /** 原始块（含 id 等 UI 态），ppl 面板统计用。 */
  getBlocksForPpl() {
    return this.blocks;
  }

  /** 持久化/请求用块（丢弃 UI 态，内容与 parse_doc 语义一致 strip；
   *  ppl 与 strip 后内容对账，避免存盘后着色对不上）。 */
  plainBlocks() {
    return this.blocks.map((b) => {
      const content = (b.content || "").trim();
      let ppl = b.ppl && b.ppl.token_texts && b.ppl.token_texts.length ? b.ppl : null;
      if (ppl) {
        if (ppl.token_texts.join("") !== content) {
          const [t, p] = reconcilePpl(ppl.token_texts, ppl.token_ppls, content);
          ppl = t.length ? { token_texts: t, token_ppls: p } : null;
        }
      }
      const out = { type: b.type, content, ppl };
      if (b.type === "cot") out.closed = !!b.closed;
      return out;
    });
  }

  /** 活动生成单元：最后一个 generate 块（与服务端契约一致）。 */
  activeBlock() {
    return ensureActive(this.blocks)[this.blocks.length - 1];
  }

  /** 活动块前紧邻的 cot 块（无则 null）。 */
  cotBlock() {
    const b = this.blocks[this.blocks.length - 2];
    return b && b.type === "cot" ? b : null;
  }

  // ---------------------------------------------------------------- 渲染

  render() {
    this.root.textContent = "";
    this.blocks.forEach((block, i) => {
      this.root.appendChild(this._renderBlock(block, i));
      this.root.appendChild(this._renderInsertZone(i + 1));
    });
  }

  _renderBlock(block, index) {
    const el = document.createElement("div");
    el.className = `block t-${block.type}`;
    if (block.type === "cot" && block.closed) el.classList.add("cot-closed");
    el.dataset.id = block.id;
    if (block.collapsed) el.classList.add("collapsed");

    const bar = document.createElement("div");
    bar.className = "block-bar";

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = TYPE_LABEL[block.type] || block.type;
    badge.title = "点击切换类型（指令↔正文）或折叠（系统/思考）";
    badge.addEventListener("click", () => this._onBadgeClick(block));
    bar.appendChild(badge);

    const tools = document.createElement("span");
    tools.className = "block-tools";
    if (block.type === "cot") {
      // 结束思考开关：closed=true → 回灌闭合思维链，模型直接写正文；
      // false → 未闭合回灌，从断点继续思考。标签不再显示在内容里。
      tools.appendChild(this._toolButton(
        block.closed ? "↩" : "⏹",
        block.closed
          ? "继续思考：重新打开思维链（下次生成从断点续写思考）"
          : "结束思考：闭合思维链（下次生成直接输出正文）",
        () => this.toggleCotClosed(block)
      ));
    }
    if (block.type === "system" || block.type === "cot") {
      tools.appendChild(this._toolButton("⤓", "展开/折叠", () => {
        block.collapsed = !block.collapsed;
        el.classList.toggle("collapsed", block.collapsed);
      }));
    }
    tools.appendChild(this._toolButton("×", "删除此块", () => {
      this.blocks = this.blocks.filter((b) => b !== block);
      ensureActive(this.blocks);
      this.opts.onChange && this.opts.onChange();
      this.render();
    }));
    bar.appendChild(tools);
    el.appendChild(bar);

    const content = document.createElement("div");
    content.className = "block-content";
    content.contentEditable = this.editable ? "true" : "false";
    if (block.type === "generate" && index === this.blocks.length - 1) {
      content.classList.add("active");
      content.dataset.ph = "续写点：光标置于此块，Ctrl+Enter 生成，F2 定稿…";
    }
    this._renderContent(block, content);

    // 编辑同步：input 只改模型；blur 时对账 ppl 并重渲染着色 span
    content.addEventListener("input", () => {
      block.content = readEditableText(content);
      this.opts.onChange && this.opts.onChange();
    });
    content.addEventListener("blur", () => {
      block.content = readEditableText(content);
      this._reconcileBlockPpl(block);
      this._renderContent(block, content);
    });
    // Enter 插入换行而非新 div；粘贴纯文本
    content.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.ctrlKey && !e.shiftKey && !e.metaKey) {
        e.preventDefault();
        document.execCommand("insertLineBreak");
      }
    });
    content.addEventListener("paste", (e) => {
      e.preventDefault();
      const text = (e.clipboardData || window.clipboardData).getData("text/plain");
      if (text) document.execCommand("insertText", false, text);
    });

    el.appendChild(content);
    return el;
  }

  /** 块内容 → 着色 span / 纯文本。generate/cot：ppl 覆盖作为内容前缀即
   *  着色该前缀，其余灰显（流式期间 ppl 滞后于文本一个 token、编辑后
   *  基线未覆盖全文等情形，避免着色整体消失/闪烁）；前缀都算不上则纯文本。
   *  （cot 只存思考正文，思维链标签不显示——closed 属性 + 按钮表达状态。） */
  _renderContent(block, el) {
    const content = block.content || "";
    el.textContent = "";
    if (block.type === "prompt" || block.type === "system") {
      el.textContent = content;
      return;
    }
    if (!content) return;
    const ppl = block.ppl;

    const covered = ppl && ppl.token_texts ? ppl.token_texts.join("") : null;
    if (covered && covered === content) {
      el.appendChild(pplSpans(ppl));
    } else if (covered && content.startsWith(covered)) {
      // 部分前缀着色：已对齐的 token 着色，其余灰显（无数据）
      el.appendChild(pplSpans(ppl));
      const rest = document.createElement("span");
      rest.style.background = NO_PPL_COLOR;
      rest.textContent = content.slice(covered.length);
      el.appendChild(rest);
    } else {
      el.textContent = content;
    }
  }

  /** 定向重渲染单个块的内容区（流式更新用，不动其他块的光标）。 */
  refreshBlock(block) {
    const el = this._blockEl(block);
    if (el) {
      const content = el.querySelector(".block-content");
      if (content) this._renderContent(block, content);
    }
  }

  _blockEl(block) {
    return this.root.querySelector(`.block[data-id="${block.id}"]`);
  }

  _toolButton(label, title, onClick) {
    const btn = document.createElement("button");
    btn.className = "tool";
    btn.textContent = label;
    btn.title = title;
    btn.type = "button";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });
    return btn;
  }

  _renderInsertZone(afterIndex) {
    const zone = document.createElement("div");
    zone.className = "insert-zone";
    const btn = document.createElement("button");
    btn.className = "insert-btn";
    btn.textContent = "+";
    btn.title = "在此插入块";
    btn.type = "button";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      this._showTypeMenu(e.clientX, e.clientY, (type) => {
        this.blocks.splice(afterIndex, 0, newBlock(type));
        ensureActive(this.blocks);
        this.opts.onChange && this.opts.onChange();
        this.render();
        const nb = this.blocks[afterIndex];
        this.focusBlock(nb);
      });
    });
    zone.appendChild(btn);
    return zone;
  }

  _showTypeMenu(x, y, onPick) {
    this._hideMenu();
    const menu = document.createElement("div");
    menu.className = "type-menu";
    for (const a of BLOCK_ACTIONS) {
      const item = document.createElement("button");
      item.type = "button";
      const name = document.createElement("span");
      name.textContent = a.label;
      const hint = document.createElement("small");
      hint.textContent = a.hint;
      item.appendChild(name);
      item.appendChild(hint);
      item.addEventListener("click", () => {
        this._hideMenu();
        onPick(a.type);
      });
      menu.appendChild(item);
    }
    document.body.appendChild(menu);
    const rect = menu.getBoundingClientRect();
    menu.style.left = Math.min(x, window.innerWidth - rect.width - 8) + "px";
    menu.style.top = Math.min(y, window.innerHeight - rect.height - 8) + "px";
    this._menu = menu;
    setTimeout(
      () => document.addEventListener("click", this._hideMenuBound = () => this._hideMenu()),
      0
    );
  }

  _hideMenu() {
    if (this._menu) {
      this._menu.remove();
      this._menu = null;
      document.removeEventListener("click", this._hideMenuBound);
    }
  }

  _onBadgeClick(block) {
    if (block.type === "prompt" || block.type === "generate") {
      block.type = block.type === "prompt" ? "generate" : "prompt";
      this.opts.onChange && this.opts.onChange();
      this.render();
    } else {
      block.collapsed = !block.collapsed;
      const el = this._blockEl(block);
      if (el) el.classList.toggle("collapsed", block.collapsed);
    }
  }

  /** 块被手动编辑后：着色对账（编辑点后合并为无数据段）。 */
  _reconcileBlockPpl(block) {
    if (!block.ppl || !block.ppl.token_texts || !block.ppl.token_texts.length) return;
    const [texts, ppls] = reconcilePpl(
      block.ppl.token_texts, block.ppl.token_ppls, block.content || ""
    );
    block.ppl = { token_texts: texts, token_ppls: ppls };
  }

  // ---------------------------------------------------------------- 编辑开关

  setEditable(editable) {
    this.editable = editable;
    for (const el of this.root.querySelectorAll(".block-content")) {
      el.contentEditable = editable ? "true" : "false";
    }
  }

  focusBlock(block, caretEnd = true) {
    const el = this._blockEl(block);
    const content = el && el.querySelector(".block-content");
    if (!content) return;
    if (block.collapsed) {
      block.collapsed = false;
      el.classList.remove("collapsed");
    }
    content.focus();
    if (caretEnd) {
      const range = document.createRange();
      range.selectNodeContents(content);
      range.collapse(false);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }

  // ---------------------------------------------------------------- 块操作

  /** 定稿：锁定当前活动块（ppl 已随块保存），末尾开启新活动块。 */
  finalizeActive() {
    const active = this.activeBlock();
    // 冻结着色对齐：与 vscode finalize 一致，strip 首尾空白 token
    this._freezePpl(active);
    const cot = this.cotBlock();
    if (cot) this._freezePpl(cot);
    this.blocks.push(newBlock("generate"));
    this.opts.onChange && this.opts.onChange();
    this.render();
    this.focusBlock(this.activeBlock());
  }

  _freezePpl(block) {
    const ppl = block.ppl;
    if (!ppl || !ppl.token_texts || !ppl.token_texts.length) return;
    const base = block.content || "";
    let [texts, ppls] = reconcilePpl(ppl.token_texts, ppl.token_ppls, base);
    while (texts.length && !texts[0].trim()) { texts.shift(); ppls.shift(); }
    while (texts.length && !texts[texts.length - 1].trim()) { texts.pop(); ppls.pop(); }
    if (texts.length) {
      texts[0] = texts[0].replace(/^\s+/, "");
      texts[texts.length - 1] = texts[texts.length - 1].replace(/\s+$/, "");
    }
    block.ppl = texts.length && texts.join("") === base ? { token_texts: texts, token_ppls: ppls } : null;
  }

  /** 顶部固化技能为 system 块。 */
  insertSkillSystemBlock(content) {
    const blk = newBlock("system", content);
    this.blocks.unshift(blk);
    this.opts.onChange && this.opts.onChange();
    this.render();
    return blk;
  }

  // ---------------------------------------------------------------- 流式更新（生成控制器调用）

  /** 确保活动块前存在 cot 块（思考增量落地目标；content=纯思考正文）。 */
  ensureCotBlock() {
    let cot = this.cotBlock();
    if (!cot) {
      cot = newBlock("cot", "");
      this.blocks.splice(this.blocks.length - 1, 0, cot);
      this.render();
    }
    return cot;
  }

  /** 思考区闭合（模型输出过闭标签）：置 closed 态，替代显示标签。 */
  markCotClosed() {
    const cot = this.cotBlock();
    if (!cot || cot.closed || !cot.content.trim()) return;
    cot.closed = true;
    this.refreshBlock(cot);
    this._refreshCotTools(cot);
  }

  /** 手动切换思考开闭（块栏按钮）。 */
  toggleCotClosed(block) {
    block.closed = !block.closed;
    this.refreshBlock(block);
    this._refreshCotTools(block);
    this.opts.onChange && this.opts.onChange();
  }

  /** 重渲染 cot 块的工具按钮（⏹/↩ 状态随 closed 切换）。 */
  _refreshCotTools(block) {
    const el = this._blockEl(block);
    if (el) el.classList.toggle("cot-closed", !!block.closed);
    const btn = el && el.querySelector(".block-tools .tool");
    if (!btn) return;
    btn.textContent = block.closed ? "↩" : "⏹";
    btn.title = block.closed
      ? "继续思考：重新打开思维链（下次生成从断点续写思考）"
      : "结束思考：闭合思维链（下次生成直接输出正文）";
  }
}

// ---------------------------------------------------------------- 工具

function pplSpans(ppl) {
  const frag = document.createDocumentFragment();
  const texts = ppl.token_texts || [];
  const ppls = ppl.token_ppls || [];
  for (let i = 0; i < texts.length; i++) {
    const span = document.createElement("span");
    const p = ppls[i];
    span.style.background = p === null || p === undefined ? NO_PPL_COLOR : pplColor(p);
    span.textContent = texts[i];
    frag.appendChild(span);
  }
  return frag;
}

/** 读取 contenteditable 文本（<br>/块级元素 → 换行）。 */
function readEditableText(el) {
  return el.innerText.replace(/\n$/, "");
}

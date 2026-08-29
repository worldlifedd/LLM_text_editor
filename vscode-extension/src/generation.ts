// 生成控制器：SSE → 流式插入、ppl 对账、定稿/锁定/追加块、状态栏与面板刷新。
import * as vscode from "vscode";
import { apiGenerate, apiSkills, apiStop, GenParams, GenUpdate, GenerateRequest } from "./api";
import { newDocumentText, parseDoc, skillSystemContent } from "./docmodel";
import { Decorator, foldingRanges } from "./decorations";
import { DocState, emptyPpl, PplSeg, reconcileActivePpl, StateStore } from "./state";

/** 面板通知消息类型。 */
export interface PplInfo {
  type: "ppl";
  ctxPpl: number | null;
  avgPpl: number | null;
  series: number[];
  cacheInfo: string;
}

export interface GenCallbacks {
  /** 更新状态栏文案 */
  statusBar(text: string, tooltip?: string): void;
  /** 重绘装饰 + 折叠 */
  refreshDecorations(): void;
  /** 向侧边面板推送消息 */
  notifyPanel(msg: unknown): void;
  /** 读取当前生成参数与启用的技能 */
  getRequestState(): { params: GenParams; context_mode: string; skills: string[] };
}

export class GenerationController {
  private ctrl: AbortController | null = null;
  private generating = false;
  private selfEdit = false;
  /** 待落地的本轮累计 ppl 段（基线 + 本轮 token）：文本插入落地 /
   * 流结束时与文档内容对账写入 activePpl，避免 SSE 快于编辑落地时
   * 误丢弃本轮 token（着色只到第一句的根因）。 */
  private pendingSeg: PplSeg | null = null;
  /** 同上，思维链段（cot 块困惑度着色）。 */
  private pendingCotSeg: PplSeg | null = null;
  private lastCacheInfo = "";

  constructor(
    private state: StateStore,
    private decorator: Decorator,
    private cb: GenCallbacks
  ) {}

  get isGenerating(): boolean {
    return this.generating;
  }

  /** 当前活动编辑器及其文档状态；无则返回 null。 */
  private session() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "markdown") return null;
    const uri = editor.document.uri.toString();
    return { editor, doc: editor.document, state: this.state.get(uri), uri };
  }

  private allPpls(parsed: ReturnType<typeof parseDoc>, ds: DocState): number[] {
    const out: number[] = [];
    for (const b of parsed.blocks) {
      if (b.type !== "generate") continue;
      const fin = ds.finalized.get(b.index);
      if (fin) {
        for (const p of fin.seg.token_ppls) if (p !== null && p !== undefined) out.push(p);
      }
    }
    for (const p of ds.activePpl.token_ppls) if (p !== null && p !== undefined) out.push(p);
    return out;
  }

  private avgPpl(ppls: number[]): number | null {
    if (!ppls.length) return null;
    let sum = 0;
    for (const p of ppls) sum += Math.log(Math.max(p, 1e-9));
    return Math.exp(sum / ppls.length);
  }

  private emitPpl(parsed: ReturnType<typeof parseDoc>, ds: DocState, cacheInfo: string): void {
    const series = this.allPpls(parsed, ds);
    this.cb.notifyPanel({
      type: "ppl",
      ctxPpl: this.ctxPpl,
      avgPpl: this.avgPpl(series),
      series,
      cacheInfo,
    } as PplInfo);
  }

  private ctxPpl: number | null = null;

  // ---------------------------------------------------------------- 生成
  async start(): Promise<void> {
    const s = this.session();
    if (!s) {
      vscode.window.showWarningMessage("请在 Markdown 文档中执行 GTE 生成");
      return;
    }
    if (this.generating) {
      vscode.window.showWarningMessage("已有生成任务进行中，请先停止（再按一次 Ctrl+Enter 即停止）");
      return;
    }
    const { editor, doc, state } = s;
    this.ctxPpl = null;
    // 等待上一轮的增量插入全部落地，再基于最新文档做对账
    await this.editQueue;

    // 确保存在活动生成单元（最后一块为 generate）
    let parsed = parseDoc(doc.getText());
    if (!parsed.active) {
      await editor.edit((e) => e.insert(new vscode.Position(doc.lineCount, 0), "\n\n<!-- generate -->\n"));
      parsed = parseDoc(doc.getText());
      if (!parsed.active) {
        vscode.window.showErrorMessage("无法创建生成块");
        return;
      }
    }

    const activeText = parsed.active.content;
    // 对账：编辑点之前的着色保留，之后灰显
    const seg = reconcileActivePpl(
      state.activePpl.token_texts,
      state.activePpl.token_ppls,
      activeText
    );
    state.activePpl = seg;
    this.pendingSeg = null;
    // 思维链基线：活动 cot 块（blocks 倒数第二块、紧邻活动生成块）已有内容对账
    const cotBlk = parsed.blocks[parsed.blocks.length - 2];
    const cotSeg =
      cotBlk && cotBlk.type === "cot"
        ? reconcileActivePpl(
            state.activeCotPpl.token_texts,
            state.activeCotPpl.token_ppls,
            cotBlk.content
          )
        : emptyPpl();
    state.activeCotPpl = cotSeg;
    this.pendingCotSeg = null;
    // 本轮基线段（覆盖 activeText 已有着色）。服务端 token_texts 为全量
    // 累计（join(token_texts) == cum_text），合并 = 基线 + 本轮累计，
    // 不能与不断膨胀的 activePpl 自身拼接（会重复累计）。
    const baseline: PplSeg = {
      token_texts: [...seg.token_texts],
      token_ppls: [...seg.token_ppls],
    };

    const { params, context_mode, skills } = this.cb.getRequestState();
    const req: GenerateRequest = {
      // 服务端契约：blocks 尾部的 generate 块即活动生成单元（parseDoc 的
      // blocks 本就以活动块结尾，直接映射即可；额外追加 active 副本会把
      // 活动内容在上下文中重复两遍）
      blocks: parsed.blocks.map((b) => ({ type: b.type, content: b.content })),
      active_text: activeText,
      skills,
      params,
      context_mode,
    };

    const ctrl = new AbortController();
    this.ctrl = ctrl;
    this.generating = true;
    this.cb.statusBar("⏳ 生成中…（再按 Ctrl+Enter 停止）");
    this.cb.notifyPanel({ type: "gen", generating: true });
    let lastCum = 0;
    let lastReasoning = 0;

    const onUpdate = (u: GenUpdate) => {
      const delta = u.cum_text.slice(lastCum);
      lastCum = u.cum_text.length;
      const rCum = u.reasoning_cum || "";
      const rDelta = rCum.slice(lastReasoning);
      lastReasoning = rCum.length;
      if (u.cache_info) this.lastCacheInfo = u.cache_info;
      if (rDelta) this.insertReasoningDelta(rDelta);
      if (delta) this.insertDelta(delta);
      // 累计着色：基线段 + 本轮全量累计 token。仅当文档内容已覆盖该段
      //（编辑队列落地）才立即对账；否则推迟到 insertDelta 落地后/流结束
      // 时补齐——若此刻对账，会把尚未落地的本轮 token 全部误判为丢失。
      const merged: PplSeg = {
        token_texts: [...baseline.token_texts, ...u.token_texts],
        token_ppls: [...baseline.token_ppls, ...u.token_ppls],
      };
      this.pendingSeg = merged;
      // 思维链段：cot 基线 + 本轮思考 token（API 无 reasoning logprobs
      // 时为空数组，思维链不着色）
      const rMerged: PplSeg = {
        token_texts: [...cotSeg.token_texts, ...(u.reasoning_token_texts || [])],
        token_ppls: [...cotSeg.token_ppls, ...(u.reasoning_token_ppls || [])],
      };
      this.pendingCotSeg = rMerged;
      const ds = this.session();
      if (ds) {
        const parsedDs = parseDoc(ds.doc.getText());
        const act = parsedDs.active;
        const cur = act ? act.content : "";
        if (cur.startsWith(merged.token_texts.join(""))) {
          ds.state.activePpl = reconcileActivePpl(merged.token_texts, merged.token_ppls, cur);
        }
        // cot 同步对账（内容已落地时；blocks 末位是活动块，cot 在倒数第二）
        const cotNow = parsedDs.blocks[parsedDs.blocks.length - 2];
        if (
          cotNow &&
          cotNow.type === "cot" &&
          cotNow.content.startsWith(rMerged.token_texts.join(""))
        ) {
          ds.state.activeCotPpl = reconcileActivePpl(
            rMerged.token_texts,
            rMerged.token_ppls,
            cotNow.content
          );
        }
      }
      const parsedNow = parseDoc(this.docText());
      if (parsedNow.active) this.emitPpl(parsedNow, this.state.get(this.currentUri()), u.cache_info || "");
      this.cb.refreshDecorations();
      if (u.final) this.finish("✅ 生成完成" + (u.cache_info ? `｜${u.cache_info}` : ""));
    };

    try {
      await apiGenerate(
        req,
        {
          onCtxPpl: (ppl) => {
            this.ctxPpl = ppl;
            const parsedNow = parseDoc(this.docText());
            if (parsedNow.active)
              this.emitPpl(parsedNow, this.state.get(this.currentUri()), "");
          },
          onUpdate,
          onError: (msg) => {
            if (ctrl.signal.aborted) return;
            this.finish(`❌ 生成失败：${msg}`, true);
          },
        },
        ctrl.signal
      );
    } catch (err: unknown) {
      const aborted = ctrl.signal.aborted || (err as Error)?.name === "AbortError";
      if (!aborted) this.finish(`❌ 生成失败：${(err as Error)?.message || err}`, true);
    } finally {
      if (this.ctrl === ctrl) this.ctrl = null;
      this.generating = false;
      // 正常完成时保留 finish 的文案（✅/❌）；仅手动停止才提示已停止
      if (ctrl.signal.aborted) this.cb.statusBar("⏹ 已停止，生成结果已保留");
      this.cb.notifyPanel({ type: "gen", generating: false });
      // 流结束：等编辑队列全部落地后做最终对账，保证尾部着色完整
      const pending = this.pendingSeg;
      this.editQueue = this.editQueue.then(() => {
        if (this.pendingSeg !== pending) return; // 已开启新一轮，交由新轮处理
        this.applyPendingSeg();
        this.applyPendingCotSeg();
        this.pendingSeg = null;
        this.pendingCotSeg = null;
        this.cb.refreshDecorations();
        const parsedNow = parseDoc(this.docText());
        if (parsedNow.active) this.emitPpl(parsedNow, this.state.get(this.currentUri()), this.lastCacheInfo);
      });
    }
  }

  private finish(msg: string, isError = false): void {
    this.cb.statusBar(msg, isError ? "GTE 生成错误" : undefined);
    // 错误必须弹窗：状态栏文案会被 10s 轮询覆盖，用户只看到"没反应"
    if (isError) vscode.window.showErrorMessage(msg);
    this.cb.notifyPanel({ type: "gen", generating: false });
  }

  private docText(): string {
    return vscode.window.activeTextEditor?.document.getText() ?? "";
  }
  private currentUri(): string {
    return vscode.window.activeTextEditor?.document.uri.toString() ?? "";
  }

  /** 串行编辑队列：SSE 事件可能快于上一次 edit 应用，直接并发计算
   * 插入点会用陈旧 offset 导致错位/乱序，故排队依次执行。 */
  private editQueue: Promise<void> = Promise.resolve();

  /** 将 pendingSeg 与当前文档内容对账后落地（在文本插入落地后调用）。 */
  private applyPendingSeg(): void {
    if (!this.pendingSeg) return;
    const s = this.session();
    if (!s) return;
    const act = parseDoc(s.doc.getText()).active;
    const cur = act ? act.content : "";
    s.state.activePpl = reconcileActivePpl(
      this.pendingSeg.token_texts,
      this.pendingSeg.token_ppls,
      cur
    );
  }

  /** 同上，思维链段：写入活动 cot 块的 ppl。 */
  private applyPendingCotSeg(): void {
    if (!this.pendingCotSeg) return;
    const s = this.session();
    if (!s) return;
    const parsed = parseDoc(s.doc.getText());
    if (!parsed.active) return;
    // blocks 末位是活动生成块，紧邻其前的 cot 即活动思维链块
    const cot = parsed.blocks[parsed.blocks.length - 2];
    if (!cot || cot.type !== "cot") return;
    s.state.activeCotPpl = reconcileActivePpl(
      this.pendingCotSeg.token_texts,
      this.pendingCotSeg.token_ppls,
      cot.content
    );
  }

  /** 在活动生成块内容末尾插入增量文本（经队列串行）。 */
  private insertDelta(delta: string): void {
    this.editQueue = this.editQueue.then(() => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "markdown") return;
      const parsed = parseDoc(editor.document.getText());
      const act = parsed.active;
      if (!act) return;
      const pos = editor.document.positionAt(act.contentEnd);
      this.selfEdit = true;
      return editor
        .edit((e) => e.insert(pos, delta), { undoStopBefore: false, undoStopAfter: false })
        .then(
          () => {
            this.selfEdit = false;
            // 文本落地后立即补齐着色（SSE 与编辑落地的时差不再丢颜色）
            this.applyPendingSeg();
            this.cb.refreshDecorations();
          },
          () => {
            this.selfEdit = false;
          }
        );
    });
  }

  /** 把思维链增量插入活动生成块前紧邻的 cot 注释块（经队列串行）。
   * 无 cot 块时先在生成块标记前创建 <!-- cot --> 块：思维链随文档保存、
   * Markdown 渲染不可见，用户可在源码中查看/编辑。 */
  private insertReasoningDelta(delta: string): void {
    this.editQueue = this.editQueue.then(async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "markdown") return;
      let parsed = parseDoc(editor.document.getText());
      const act = parsed.active;
      if (!act) return;
      // 活动块是 blocks 的最后一块，其前一块即 cot 候选
      let cot = parsed.blocks[parsed.blocks.length - 2];
      if (!cot || cot.type !== "cot") {
        this.selfEdit = true;
        try {
          const pos = editor.document.positionAt(act.blockStart);
          await editor.edit(
            (e) => e.insert(pos, "<!-- cot\n-->\n\n"),
            { undoStopBefore: false, undoStopAfter: false }
          );
        } finally {
          this.selfEdit = false;
        }
        parsed = parseDoc(editor.document.getText());
        cot = parsed.blocks[parsed.blocks.length - 2];
        if (!cot || cot.type !== "cot") return;
      }
      this.selfEdit = true;
      try {
        const pos = editor.document.positionAt(cot.contentEnd);
        await editor.edit(
          (e) => e.insert(pos, delta),
          { undoStopBefore: false, undoStopAfter: false }
        );
      } finally {
        this.selfEdit = false;
      }
      // 思维链文本落地后立即补齐着色（与正文 insertDelta 同思路）
      this.applyPendingCotSeg();
      this.cb.refreshDecorations();
    });
  }

  async stop(): Promise<void> {
    this.cb.statusBar("⏹ 正在停止…");
    try {
      await apiStop();
    } catch {
      /* 服务不可达也继续中止本地流 */
    }
    this.ctrl?.abort();
  }

  // ---------------------------------------------------------------- 定稿 / 追加 / 锁定
  async finalize(): Promise<void> {
    const s = this.session();
    if (!s) return;
    // 等待流式插入全部落地再读取，否则冻结的着色可能缺尾部
    await this.editQueue;
    const { editor, doc, state } = s;
    const parsed = parseDoc(doc.getText());
    if (!parsed.active) {
      vscode.window.showWarningMessage("当前文档没有活动生成块");
      return;
    }
    // 冻结着色：strip 首尾空白后与块内容对齐
    const texts = [...state.activePpl.token_texts];
    const ppls = [...state.activePpl.token_ppls];
    while (texts.length && !texts[0].trim()) {
      texts.shift();
      ppls.shift();
    }
    while (texts.length && !texts[texts.length - 1].trim()) {
      texts.pop();
      ppls.pop();
    }
    if (texts.length) {
      texts[0] = texts[0].replace(/^\s+/, "");
      texts[texts.length - 1] = texts[texts.length - 1].replace(/\s+$/, "");
    }
    if (texts.length && "".concat(...texts) === parsed.active.content) {
      state.finalized.set(parsed.active.index, {
        seg: { token_texts: texts, token_ppls: ppls },
        content: parsed.active.content,
      });
    }
    // 思维链块冻结：活动 cot（blocks 倒数第二块）同样定稿着色
    const cotBlk = parsed.blocks[parsed.blocks.length - 2];
    if (
      cotBlk &&
      cotBlk.type === "cot" &&
      state.activeCotPpl.token_texts.length &&
      "".concat(...state.activeCotPpl.token_texts) === cotBlk.content
    ) {
      state.finalized.set(cotBlk.index, {
        seg: {
          token_texts: [...state.activeCotPpl.token_texts],
          token_ppls: [...state.activeCotPpl.token_ppls],
        },
        content: cotBlk.content,
      });
    }
    // 锁定前序全部块，聚焦新的活动块
    for (const b of parsed.blocks) state.locked.add(b.index);
    state.activePpl = emptyPpl();
    state.activeCotPpl = emptyPpl();

    await editor.edit((e) => e.insert(new vscode.Position(doc.lineCount, 0), "\n\n<!-- generate -->\n"));
    this.cb.statusBar("已定稿当前生成块并开启新块（前序块已锁定折叠）");
    this.cb.refreshDecorations();
  }

  async addPrompt(): Promise<void> {
    const s = this.session();
    if (!s) return;
    await s.editor.edit((e) =>
      e.insert(new vscode.Position(s.doc.lineCount, 0), "\n\n<!-- prompt\n-->")
    );
    this.cb.statusBar("已追加提示词块");
    this.cb.refreshDecorations();
  }

  /** 选择技能并固化为文档顶部的 system 块：文档自包含，
   * 换到没有该技能的环境也能复现生成过程。 */
  async insertSkillBlock(): Promise<void> {
    const s = this.session();
    if (!s) {
      vscode.window.showWarningMessage("请在 Markdown 文档中固化技能");
      return;
    }
    let skills;
    try {
      skills = await apiSkills();
    } catch (e) {
      vscode.window.showErrorMessage(`获取技能库失败：${(e as Error).message}`);
      return;
    }
    if (!skills.length) {
      vscode.window.showInformationMessage("技能库为空（服务端 skills/ 目录无技能）");
      return;
    }
    const pick = await vscode.window.showQuickPick(
      skills.map((sk) => ({ label: sk.name, description: sk.description, skill: sk })),
      { placeHolder: "选择要固化为系统提示词块的技能" }
    );
    if (!pick) return;
    const content = skillSystemContent(pick.skill);
    await s.editor.edit((e) =>
      e.insert(new vscode.Position(0, 0), `<!-- system\n${content}\n-->\n\n`)
    );
    this.cb.statusBar(`已固化技能「${pick.label}」为系统提示词块（文档自包含）`);
    this.cb.refreshDecorations();
  }

  toggleLock(): void {
    const s = this.session();
    if (!s) return;
    const parsed = parseDoc(s.doc.getText());
    const cursor = s.doc.offsetAt(s.editor.selection.active);
    const blk = parsed.blocks.find((b) => cursor >= b.blockStart && cursor <= b.blockEnd);
    if (!blk) {
      vscode.window.showWarningMessage("光标不在任何块内");
      return;
    }
    if (s.state.locked.has(blk.index)) s.state.locked.delete(blk.index);
    else s.state.locked.add(blk.index);
    this.cb.refreshDecorations();
  }

  // ---------------------------------------------------------------- 编辑同步
  /** 文档内容变化：用户编辑时对账 ppl 并重绘。 */
  onDocChanged(e: vscode.TextDocumentChangeEvent): void {
    if (this.selfEdit) return;
    if (e.document.languageId !== "markdown") return;
    const uri = e.document.uri.toString();
    const ds = this.state.get(uri);
    const parsed = parseDoc(e.document.getText());
    const act = parsed.active;
    if (act) {
      ds.activePpl = reconcileActivePpl(
        ds.activePpl.token_texts,
        ds.activePpl.token_ppls,
        act.content
      );
    }
    this.cb.refreshDecorations();
  }

  /** 编辑器切换/滚动后重绘。 */
  refresh(): void {
    this.cb.refreshDecorations();
  }
}

export { newDocumentText };

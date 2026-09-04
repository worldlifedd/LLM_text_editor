// 块模型核心（可移植内核：不依赖 DOM，仅纯数据与纯函数）。
// 与 Python 端 core.py 的 ppl 对账、着色、消息组装语义保持一致。
// ---------------------------------------------------------------- 类型

export const BLOCK_TYPES = ["prompt", "system", "cot", "generate"];

export const TYPE_LABEL = {
  prompt: "指令",
  system: "系统",
  cot: "思考",
  generate: "正文",
};

/** 新块。id 仅前端渲染用，持久化时可丢弃。 */
export function newBlock(type, content = "") {
  const b = {
    id: "b" + Math.random().toString(36).slice(2, 10),
    type,
    content: content || "",
    ppl: null, // { token_texts: string[], token_ppls: (number|null)[] }
    collapsed: type === "cot" || type === "system", // 折叠仅为 UI 态
  };
  if (type === "cot") b.closed = false; // 思考是否已结束（按钮切换）
  return b;
}

/** 文档不变式：末块为 generate（活动单元）。不满足则补空块。 */
export function ensureActive(blocks) {
  if (!blocks.length || blocks[blocks.length - 1].type !== "generate") {
    blocks.push(newBlock("generate"));
  }
  return blocks;
}

// ---------------------------------------------------------------- 困惑度着色
// 与 core.ppl_rgb / ppl_color 一致：log(ppl) 在 [0, log200] clamp，
// 绿→黄→红 三锚点插值。

const LOG_PPL_MAX = Math.log(200.0);
const ANCHORS = [
  [134, 226, 148],
  [255, 226, 130],
  [255, 118, 108],
]; // 绿 → 黄 → 红

export function pplColor(ppl) {
  let t = (Math.log(Math.max(ppl, 1.0001)) - 0.0) / LOG_PPL_MAX;
  t = Math.min(Math.max(t, 0.0), 1.0);
  let a, b, u;
  if (t < 0.5) {
    [a, b, u] = [ANCHORS[0], ANCHORS[1], t * 2];
  } else {
    [a, b, u] = [ANCHORS[1], ANCHORS[2], (t - 0.5) * 2];
  }
  const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * u));
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.45)`;
}

export const NO_PPL_COLOR = "rgba(160,160,160,0.18)";

/**
 * 着色对账（移植 core.reconcile_active_ppl）：块被手动编辑后，仅保留
 * 与新文本公共字符前缀内完整 token 的着色；编辑点及之后的旧文本合并为
 * 单个无数据段(ppl=null)。返回 (token_texts, token_ppls) 覆盖整个 base。
 */
export function reconcilePpl(tokenTexts, tokenPpls, base) {
  if (!tokenTexts || !tokenTexts.length) return [[], []];
  const cov = tokenTexts.join("");
  if (base.startsWith(cov)) {
    // 覆盖是 base 前缀但短于 base（如末尾追加文本）：补无数据段，
    // 保证覆盖整个 base——否则与后续段拼接会出现字符空洞
    const rest = base.slice(cov.length);
    if (!rest) return [[...tokenTexts], [...tokenPpls]];
    return [[...tokenTexts, rest], [...tokenPpls, null]];
  }
  const n = Math.min(cov.length, base.length);
  let k = 0;
  while (k < n && cov[k] === base[k]) k++;
  const keptT = [], keptP = [];
  let acc = 0;
  for (let i = 0; i < tokenTexts.length; i++) {
    const t = tokenTexts[i];
    if (acc + t.length <= k) {
      keptT.push(t);
      keptP.push(tokenPpls[i]);
      acc += t.length;
    } else break;
  }
  const rest = base.slice(acc);
  if (rest) {
    keptT.push(rest);
    keptP.push(null);
  }
  return [keptT, keptP];
}

/** 空着色段。 */
export function emptyPpl() {
  return { token_texts: [], token_ppls: [] };
}

/**
 * prefill 打分合并：服务端对活动块文本（含手动编辑部分）的逐 token ppl
 * （prefill_ppl 事件）覆盖 activeText 的一个后缀，与既有基线合并——
 * 公共前缀（缓存命中的未编辑部分）保留旧着色，prefill 覆盖的尾部替换为
 * 新鲜打分，编辑点交叉 token / 无数据区域补灰段。返回新基线；数据
 * 无法对齐（join 非 activeText 后缀）时返回 null。
 */
export function mergePrefillPpl(baseline, activeText, prefill) {
  const pt = (prefill && prefill.token_texts) || [];
  const pp = (prefill && prefill.token_ppls) || [];
  const covered = pt.join("");
  if (!covered || !activeText.endsWith(covered)) return null;
  const head = activeText.slice(0, activeText.length - covered.length);
  const [ht, hp] = reconcilePpl(
    baseline.token_texts, baseline.token_ppls, head
  );
  const hCov = ht.join("");
  if (hCov.length < head.length) {
    // 基线无数据 / 跨编辑点 token 被对账丢弃 → 补灰段，
    // 保证合并后覆盖 activeText 全文（否则出现字符空洞）
    ht.push(head.slice(hCov.length));
    hp.push(null);
  }
  return { token_texts: [...ht, ...pt], token_ppls: [...hp, ...pp] };
}

/** 几何平均困惑度；无数据返回 null。 */
export function avgPpl(ppls) {
  const vals = (ppls || []).filter((p) => p !== null && p !== undefined);
  if (!vals.length) return null;
  let sum = 0;
  for (const p of vals) sum += Math.log(Math.max(p, 1e-9));
  return Math.exp(sum / vals.length);
}

// ---------------------------------------------------------------- 思维链
// 标签拼接构造，避免字面量被工具链按 HTML 清洗（同 backend.py / docmodel.ts）
export const THINK_OPEN = "<" + "thi" + "nk>";
export const THINK_CLOSE = "</" + "thi" + "nk>";

/**
 * 提取 cot 块纯思考文本（旧格式遗留：去掉标签与"Thought for ... seconds"
 * 前缀）。新格式 content 本身就是纯思考正文，加载时经 normalizeCotBlock
 * 归一化，运行期不再调用。
 */
export function cotBody(content) {
  let s = content || "";
  const oi = s.indexOf(THINK_OPEN);
  if (oi >= 0) s = s.slice(oi + THINK_OPEN.length);
  if (s.startsWith("\n")) s = s.slice(1);
  const ci = s.lastIndexOf(THINK_CLOSE);
  if (ci >= 0) s = s.slice(0, ci);
  if (s.endsWith("\n")) s = s.slice(0, -1);
  return s;
}

/**
 * 旧格式 cot 块归一化（加载历史文档用）：标签写在 content 里 →
 * content=纯思考正文 + closed 属性。新格式块原样通过。
 */
export function normalizeCotBlock(block) {
  if (block.type !== "cot") return block;
  if (block.closed === undefined && (block.content || "").includes(THINK_OPEN)) {
    block.closed = (block.content || "").includes(THINK_CLOSE);
    block.content = cotBody(block.content);
  }
  if (block.closed === undefined) block.closed = true; // 纯文本历史块按已结束处理
  return block;
}

// ---------------------------------------------------------------- LLM 视角
/**
 * 块序列 → chat messages（移植 core.build_prompt 的 chat 模式语义，
 * 不含模板包装，用于"LLM 视角预览"）：
 * - system 块 + 本地技能 → system 消息（在前）
 * - prompt → user、generate → assistant，相邻同角色合并
 * - cot 不进上下文；唯一例外：正文为空且紧邻活动块的 cot 是"被中断的
 *   思考"，作为续写头（assistant 前缀）回灌
 * - 活动块（末位 generate）= assistant 续写前缀
 */
export function blocksToMessages(blocks, skillCtx) {
  const msgs = [];
  if (skillCtx) msgs.push({ role: "system", content: skillCtx });
  for (const b of blocks || []) {
    const c = (b.content || "").trim();
    if (!c) continue;
    if (b.type === "system") {
      if (msgs.length && msgs[0].role === "system") {
        msgs[0].content += "\n\n" + c;
      } else {
        msgs.unshift({ role: "system", content: c });
      }
    } else if (b.type === "cot") {
      continue;
    } else {
      const role = b.type === "prompt" ? "user" : "assistant";
      if (msgs.length && msgs[msgs.length - 1].role === role) {
        msgs[msgs.length - 1].content += "\n\n" + c;
      } else {
        msgs.push({ role, content: c });
      }
    }
  }
  return msgs;
}

/** 活动块前紧邻的 cot（当前进行中回复的思考区，恒回灌；与 core.build_prompt 一致）。 */
export function pendingCot(blocks) {
  const active = blocks[blocks.length - 1];
  if (!active || active.type !== "generate") return "";
  for (let i = blocks.length - 2; i >= 0; i--) {
    const b = blocks[i];
    if (b.type === "cot") return (b.content || "").trim();
    if (b.type !== "generate") break; // 隔着 prompt/system，已不是本轮思考
  }
  return "";
}

// ---------------------------------------------------------------- 技能
/** 技能 → system 块内容（与后端 skills_to_context 单技能格式一致）。 */
export function skillSystemContent(skill) {
  let header = `# 技能指令: ${skill.name}`;
  if (skill.description) header += `\n# 说明: ${skill.description}`;
  return `${header}\n${skill.instructions}`;
}

/** 默认新文档块。 */
export function newDocumentBlocks() {
  return ensureActive([
    newBlock("prompt", "写一段关于秋天的散文，100字左右。"),
  ]);
}

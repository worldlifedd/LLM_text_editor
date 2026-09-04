// 每文档内存状态：活动生成单元 ppl、定稿块 ppl 冻结、锁定块集合。

export interface PplSeg {
  token_texts: string[];
  token_ppls: (number | null)[];
}

export const emptyPpl = (): PplSeg => ({ token_texts: [], token_ppls: [] });

/** 定稿块冻结的 ppl：key = 块 index，值同时记录内容以校验失配。 */
export interface FinalizedPpl {
  seg: PplSeg;
  content: string;
}

export interface DocState {
  activePpl: PplSeg;
  /** 活动思维链块（紧邻活动生成块的 cot 注释）的 ppl——本地后端思考
   * token 的 log-prob 与正文同源，思维链同样可困惑度着色 */
  activeCotPpl: PplSeg;
  finalized: Map<number, FinalizedPpl>;
  locked: Set<number>;
}

function newState(): DocState {
  return {
    activePpl: emptyPpl(),
    activeCotPpl: emptyPpl(),
    finalized: new Map(),
    locked: new Set(),
  };
}

export class StateStore {
  private map = new Map<string, DocState>();

  get(uri: string): DocState {
    let s = this.map.get(uri);
    if (!s) {
      s = newState();
      this.map.set(uri, s);
    }
    return s;
  }

  clear(uri: string): void {
    this.map.delete(uri);
  }
}

// ---- ppl 对账（TS 移植 core.reconcile_active_ppl）----
export function reconcileActivePpl(
  tokenTexts: string[],
  tokenPpls: (number | null)[],
  base: string
): PplSeg {
  if (!tokenTexts.length) return emptyPpl();
  const cov = tokenTexts.join("");
  if (base.startsWith(cov)) {
    // 覆盖是 base 前缀但短于 base（如末尾追加文本）：补无数据段，
    // 保证覆盖整个 base——否则与后续段拼接会出现字符空洞
    const rest = base.slice(cov.length);
    if (!rest) return { token_texts: [...tokenTexts], token_ppls: [...tokenPpls] };
    return {
      token_texts: [...tokenTexts, rest],
      token_ppls: [...tokenPpls, null],
    };
  }
  let k = 0;
  const n = Math.min(cov.length, base.length);
  while (k < n && cov[k] === base[k]) k++;
  const keptT: string[] = [];
  const keptP: (number | null)[] = [];
  let acc = 0;
  for (let i = 0; i < tokenTexts.length; i++) {
    const t = tokenTexts[i];
    if (acc + t.length <= k) {
      keptT.push(t);
      keptP.push(tokenPpls[i]);
      acc += t.length;
    } else {
      break;
    }
  }
  const rest = base.slice(acc);
  if (rest) {
    keptT.push(rest);
    keptP.push(null);
  }
  return { token_texts: keptT, token_ppls: keptP };
}

/**
 * prefill 打分合并（TS 移植 web/model.js mergePrefillPpl）：服务端对活动块
 * 文本（含手动编辑部分）的逐 token ppl（prefill_ppl 事件）覆盖 activeText
 * 的一个后缀，与既有基线合并——公共前缀（缓存命中的未编辑部分）保留旧
 * 着色，prefill 覆盖的尾部替换为新鲜打分，编辑点交叉 token / 无数据区域
 * 补灰段。返回新基线；数据无法对齐时返回 null。
 */
export function mergePrefillPpl(
  baseline: PplSeg,
  activeText: string,
  prefill: { token_texts: string[]; token_ppls: (number | null)[] }
): PplSeg | null {
  const pt = (prefill && prefill.token_texts) || [];
  const pp = (prefill && prefill.token_ppls) || [];
  const covered = "".concat(...pt);
  if (!covered || !activeText.endsWith(covered)) return null;
  const head = activeText.slice(0, activeText.length - covered.length);
  const hSeg = reconcileActivePpl(baseline.token_texts, baseline.token_ppls, head);
  const hCov = "".concat(...hSeg.token_texts);
  if (hCov.length < head.length) {
    // 基线无数据 / 跨编辑点 token 被对账丢弃 → 补灰段，
    // 保证合并后覆盖 activeText 全文（否则出现字符空洞）
    hSeg.token_texts.push(head.slice(hCov.length));
    hSeg.token_ppls.push(null);
  }
  return {
    token_texts: [...hSeg.token_texts, ...pt],
    token_ppls: [...hSeg.token_ppls, ...pp],
  };
}

/**
 * 把 ppl 段补齐到 base 长度：不足部分补一个 null 段（不着色，但保持对齐）。
 *
 * 续写时若上一轮的对账被跳过（流被中断、新一轮抢先开始），state 里的段会
 * 短于 cot 块内容。不补齐的话，reconcileActivePpl 会把整块思维链压成单个
 * null 段，本轮新增 token 也跟着对不上，表现为"思维链完全不着色"。
 * 补齐后至少保证已有着色不丢、本轮新增正常着色。
 */
export function padSegTo(seg: PplSeg, base: string): PplSeg {
  const cov = "".concat(...seg.token_texts);
  if (base.length <= cov.length) return seg;
  const rest = base.slice(cov.length);
  if (!rest) return seg;
  return {
    token_texts: [...seg.token_texts, rest],
    token_ppls: [...seg.token_ppls, null],
  };
}

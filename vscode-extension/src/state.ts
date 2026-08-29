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
    return { token_texts: [...tokenTexts], token_ppls: [...tokenPpls] };
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

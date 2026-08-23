// 编辑器装饰：块背景着色、困惑度热力图、锁定灰显。TS 移植 core.ppl_rgb。
import * as vscode from "vscode";
import { ParsedDoc } from "./docmodel";
import { DocState } from "./state";

const LOG_PPL_MAX = Math.log(200);
export function pplT(ppl: number): number {
  const t = (Math.log(Math.max(ppl, 1.0001)) - 0) / LOG_PPL_MAX;
  return Math.min(Math.max(t, 0), 1);
}
export function pplRgb(ppl: number): [number, number, number] {
  const t = pplT(ppl);
  const anchors: [number, number, number][] = [
    [134, 226, 148],
    [255, 226, 130],
    [255, 118, 108],
  ];
  let a: [number, number, number], b: [number, number, number], u: number;
  if (t < 0.5) {
    a = anchors[0];
    b = anchors[1];
    u = t * 2;
  } else {
    a = anchors[1];
    b = anchors[2];
    u = (t - 0.5) * 2;
  }
  return [
    Math.round(a[0] + (b[0] - a[0]) * u),
    Math.round(a[1] + (b[1] - a[1]) * u),
    Math.round(a[2] + (b[2] - a[2]) * u),
  ];
}

/** 热力图按 log 尺度分桶，每桶一个固定背景色（避免每 token 一个 decoration type）。 */
const HEAT_BUCKETS = 16;
function bucketColor(ppl: number): string {
  const idx = Math.min(HEAT_BUCKETS - 1, Math.floor(pplT(ppl) * HEAT_BUCKETS));
  const midT = (idx + 0.5) / HEAT_BUCKETS;
  const pplAt = Math.exp(midT * LOG_PPL_MAX);
  const [r, g, b] = pplRgb(pplAt);
  return `rgba(${r},${g},${b},0.45)`;
}

const GRAY_BG = "rgba(160,160,160,0.18)";

export class Decorator {
  private heatTypes = new Map<string, vscode.TextEditorDecorationType>();
  private grayType = vscode.window.createTextEditorDecorationType({
    backgroundColor: GRAY_BG,
  });
  private promptBg = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(110,150,255,0.10)",
  });
  private generateBg = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(60,190,120,0.10)",
  });
  private lockedBg = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(160,160,160,0.12)",
    before: { contentText: "🔒 ", margin: "0 2px 0 0" },
  });

  private heatType(color: string): vscode.TextEditorDecorationType {
    let t = this.heatTypes.get(color);
    if (!t) {
      t = vscode.window.createTextEditorDecorationType({
        backgroundColor: color,
      });
      this.heatTypes.set(color, t);
    }
    return t;
  }

  /** 将 token 段映射为热力/灰显装饰（从 startOffset 起）。 */
  private tokenSegs(
    doc: vscode.TextDocument,
    tokenTexts: string[],
    tokenPpls: (number | null)[],
    startOffset: number,
    out: Map<vscode.TextEditorDecorationType, vscode.Range[]>
  ): void {
    let off = startOffset;
    for (let i = 0; i < tokenTexts.length; i++) {
      const t = tokenTexts[i];
      if (!t) continue;
      const end = off + t.length;
      const range = new vscode.Range(doc.positionAt(off), doc.positionAt(end));
      const p = tokenPpls[i];
      const type =
        p === null || p === undefined ? this.grayType : this.heatType(bucketColor(p));
      let arr = out.get(type);
      if (!arr) {
        arr = [];
        out.set(type, arr);
      }
      arr.push(range);
      off = end;
    }
  }

  /** 重绘整个文档的装饰。 */
  apply(editor: vscode.TextEditor, parsed: ParsedDoc, state: DocState): void {
    const doc = editor.document;
    const heat: Map<vscode.TextEditorDecorationType, vscode.Range[]> = new Map();
    const prompts: vscode.Range[] = [];
    const generates: vscode.Range[] = [];
    const locked: vscode.Range[] = [];

    for (const blk of parsed.blocks) {
      const range = new vscode.Range(
        doc.positionAt(blk.contentStart),
        doc.positionAt(blk.contentEnd)
      );
      if (blk.type === "prompt") {
        prompts.push(range);
      } else {
        generates.push(range);
        // 定稿块困惑度：校验内容匹配后按 token 着色，失配整块灰显
        const fin = state.finalized.get(blk.index);
        const segOk =
          fin && fin.seg.token_texts.length &&
          "".concat(...fin.seg.token_texts) === blk.content;
        if (segOk) {
          this.tokenSegs(doc, fin!.seg.token_texts, fin!.seg.token_ppls, blk.contentStart, heat);
        } else {
          this.tokenSegs(doc, [blk.content], [null], blk.contentStart, heat);
        }
      }
      if (state.locked.has(blk.index)) {
        locked.push(range);
      }
    }

    // 活动生成单元：activePpl 覆盖其内容
    const active = parsed.active;
    if (active) {
      generates.push(
        new vscode.Range(
          doc.positionAt(active.contentStart),
          doc.positionAt(active.contentEnd)
        )
      );
      if (state.activePpl.token_texts.length) {
        this.tokenSegs(
          doc,
          state.activePpl.token_texts,
          state.activePpl.token_ppls,
          active.contentStart,
          heat
        );
      }
    }

    for (const [type, ranges] of heat) editor.setDecorations(type, ranges);
    editor.setDecorations(this.grayType, []);
    editor.setDecorations(this.promptBg, prompts);
    editor.setDecorations(this.generateBg, generates);
    editor.setDecorations(this.lockedBg, locked);
  }

  dispose(): void {
    for (const t of this.heatTypes.values()) t.dispose();
    this.heatTypes.clear();
    this.grayType.dispose();
    this.promptBg.dispose();
    this.generateBg.dispose();
    this.lockedBg.dispose();
  }
}

/** 锁定块折叠范围：整块折叠（start 行 ~ close 标签所在行）。 */
export function foldingRanges(
  parsed: ParsedDoc,
  doc: vscode.TextDocument
): vscode.FoldingRange[] {
  return parsed.blocks.map((b) => {
    const startLine = doc.positionAt(b.blockStart).line;
    let endLine = doc.positionAt(b.blockEnd).line;
    const endPos = doc.positionAt(b.blockEnd);
    if (endPos.character === 0 && endLine > startLine) endLine -= 1;
    return new vscode.FoldingRange(startLine, endLine);
  });
}

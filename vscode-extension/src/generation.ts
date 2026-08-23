// 生成控制器：SSE → 流式插入、ppl 对账、定稿/锁定/追加块、状态栏与面板刷新。
import * as vscode from "vscode";
import { apiGenerate, apiStop, GenParams, GenUpdate, GenerateRequest } from "./api";
import { newDocumentText, parseDoc } from "./docmodel";
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
      vscode.window.showWarningMessage("已有生成任务进行中，请先停止");
      return;
    }
    const { editor, doc, state } = s;
    this.ctxPpl = null;

    // 确保存在活动生成单元（最后一块为 generate）
    let parsed = parseDoc(doc.getText());
    if (!parsed.active) {
      await editor.edit((e) => e.insert(new vscode.Position(doc.lineCount, 0), "\n\n<generate>\n\n</generate>"));
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

    const { params, context_mode, skills } = this.cb.getRequestState();
    const req: GenerateRequest = {
      blocks: parsed.blocks.map((b) => ({ type: b.type, content: b.content })),
      active_text: activeText,
      skills,
      params,
      context_mode,
    };

    const ctrl = new AbortController();
    this.ctrl = ctrl;
    this.generating = true;
    this.cb.statusBar("⏳ 生成中…（Ctrl+Alt+Enter 停止）");
    let lastCum = 0;

    const onUpdate = (u: GenUpdate) => {
      const delta = u.cum_text.slice(lastCum);
      lastCum = u.cum_text.length;
      if (delta) this.insertDelta(delta);
      // 累计着色：基线段 + 本轮新 token，随后按当前内容对账
      const ds = this.session();
      if (ds) {
        const merged: PplSeg = {
          token_texts: [...ds.state.activePpl.token_texts, ...u.token_texts],
          token_ppls: [...ds.state.activePpl.token_ppls, ...u.token_ppls],
        };
        const act = parseDoc(ds.doc.getText()).active;
        const cur = act ? act.content : "";
        ds.state.activePpl = reconcileActivePpl(merged.token_texts, merged.token_ppls, cur);
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
      if (!ctrl.signal.aborted) this.cb.statusBar("⏹ 已停止，生成结果已保留");
      this.cb.notifyPanel({ type: "gen", generating: false });
      this.cb.refreshDecorations();
    }
  }

  private finish(msg: string, isError = false): void {
    this.cb.statusBar(msg, isError ? "GTE 生成错误" : undefined);
    this.cb.notifyPanel({ type: "gen", generating: false });
  }

  private docText(): string {
    return vscode.window.activeTextEditor?.document.getText() ?? "";
  }
  private currentUri(): string {
    return vscode.window.activeTextEditor?.document.uri.toString() ?? "";
  }

  /** 在活动生成块内容末尾插入增量文本。 */
  private insertDelta(delta: string): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const parsed = parseDoc(editor.document.getText());
    const act = parsed.active;
    if (!act) return;
    const pos = editor.document.positionAt(act.contentEnd);
    this.selfEdit = true;
    void editor
      .edit((e) => e.insert(pos, delta), { undoStopBefore: false, undoStopAfter: false })
      .then(() => {
        this.selfEdit = false;
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
    // 锁定前序全部块，聚焦新的活动块
    for (const b of parsed.blocks) state.locked.add(b.index);
    state.activePpl = emptyPpl();

    await editor.edit((e) => e.insert(new vscode.Position(doc.lineCount, 0), "\n\n<generate>\n\n</generate>"));
    this.cb.statusBar("已定稿当前生成块并开启新块（前序块已锁定折叠）");
    this.cb.refreshDecorations();
  }

  async addPrompt(): Promise<void> {
    const s = this.session();
    if (!s) return;
    await s.editor.edit((e) =>
      e.insert(new vscode.Position(s.doc.lineCount, 0), "\n\n<prompt>\n\n</prompt>")
    );
    this.cb.statusBar("已追加提示词块");
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

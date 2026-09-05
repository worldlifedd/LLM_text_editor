// Web 编辑器面板：以 WebviewPanel 承载 web/ 共用块编辑器前端（编辑器标签页）。
// HTML 标记直接复用 web/index.html（esbuild 以 text 方式内联），脚本/样式为
// dist/webview.js + dist/webview.css（web/webview-main.js 打包产物）。
// webview 不直连网络：所有 HTTP（含 SSE 生成流）经本文件 postMessage 代理。
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import indexHtml from "../../web/index.html";
import {
  apiStatus, apiLoad, apiSkills, apiStop, apiGenerate,
  apiListDocs, apiGetDoc, apiSaveDoc, apiDeleteDoc, apiImportMarkdown,
  fetchExportMd, GenerateRequest,
} from "./api";
import { ensureServer } from "./server";

const SETTINGS_KEY = "gte.webview.settings";
const DRAFT_KEY = "gte.webview.draft";

interface ReqMessage {
  type: "req";
  id: number;
  method: string;
  args: Record<string, unknown>;
}

interface AbortMessage {
  type: "abort";
  id: number;
}

export class WebEditorPanel {
  public static current: WebEditorPanel | null = null;

  public static createOrShow(context: vscode.ExtensionContext): void {
    if (WebEditorPanel.current) {
      WebEditorPanel.current.panel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "gte.webEditor",
      "GTE 生成式文本编辑器",
      vscode.ViewColumn.Active,
      {
        enableScripts: true,
        // 切换标签页时保持 webview 存活：SSE 生成流与编辑状态不中断
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, "dist")],
      }
    );
    WebEditorPanel.current = new WebEditorPanel(panel, context);
  }

  private readonly genControllers = new Map<number, AbortController>();

  private constructor(
    private readonly panel: vscode.WebviewPanel,
    private readonly context: vscode.ExtensionContext
  ) {
    panel.webview.html = this.makeHtml(panel.webview, context.extensionUri);
    panel.webview.onDidReceiveMessage((m) => void this.handle(m));
    panel.onDidDispose(() => {
      WebEditorPanel.current = null;
      this.genControllers.clear();
    });
  }

  private post(msg: unknown): void {
    void this.panel.webview.postMessage(msg);
  }

  // ------------------------------------------------------------- 消息代理

  /** 需要代理到 server.py 的方法（其余为本地能力：设置/草稿/对话框）。 */
  private static readonly SERVER_METHODS = new Set([
    "status", "load", "skills", "stop", "generate",
    "docs.list", "docs.get", "docs.save", "docs.delete",
    "docs.importMd", "docs.exportMd",
  ]);

  private async handle(raw: unknown): Promise<void> {
    const m = raw as ReqMessage | AbortMessage;
    if (!m || typeof m !== "object") return;
    if (m.type === "abort") {
      this.genControllers.get(m.id)?.abort();
      return;
    }
    if (m.type !== "req" || typeof m.id !== "number") return;

    const { id, method } = m;
    const args = m.args || {};
    if (WebEditorPanel.SERVER_METHODS.has(method)) {
      try {
        await ensureServer(); // 代理请求前确保服务可达（必要时自动拉起）
      } catch (e) {
        this.post({ type: "resp", id, ok: false, error: (e as Error).message });
        return;
      }
    }
    try {
      const data = await this.dispatch(id, method, args);
      this.post({ type: "resp", id, ok: true, data: data === undefined ? null : data });
    } catch (e) {
      this.post({
        type: "resp",
        id,
        ok: false,
        error: (e as Error).message || String(e),
      });
    }
  }

  private async dispatch(
    id: number,
    method: string,
    args: Record<string, unknown>
  ): Promise<unknown> {
    switch (method) {
      case "status":
        return apiStatus();
      case "load":
        return apiLoad(args as Parameters<typeof apiLoad>[0]);
      case "skills":
        return apiSkills();
      case "stop":
        return apiStop();
      case "generate":
        return this.proxyGenerate(id, (args as { req: GenerateRequest }).req);
      // ------------------------------------------------------------ 文档
      case "docs.list":
        return apiListDocs();
      case "docs.get":
        return apiGetDoc(String(args.id));
      case "docs.save":
        return apiSaveDoc(args as Parameters<typeof apiSaveDoc>[0]);
      case "docs.delete":
        return apiDeleteDoc(String(args.id));
      case "docs.importMd":
        return apiImportMarkdown(String(args.text), String(args.title));
      case "docs.exportMd":
        return this.exportMd(String(args.id), String(args.title));
      case "docs.pickFile":
        return this.pickMdFile();
      // ------------------------------------------------------------ 平台能力
      case "confirm": {
        const pick = await vscode.window.showInformationMessage(
          String(args.message || ""),
          { modal: true },
          "确定"
        );
        return pick === "确定";
      }
      // webview 生命周期不可靠（dispose 后 state 丢失）→ globalState 持久化
      case "settings.get":
        return this.context.globalState.get(SETTINGS_KEY) ?? null;
      case "settings.set":
        await this.context.globalState.update(SETTINGS_KEY, args.value ?? undefined);
        return null;
      case "draft.get":
        return this.context.globalState.get(DRAFT_KEY) ?? null;
      case "draft.set":
        await this.context.globalState.update(DRAFT_KEY, args.value ?? undefined);
        return null;
      default:
        throw new Error(`未知方法：${method}`);
    }
  }

  /** 生成流代理：SSE 事件逐个转发，AbortController 支持停止。 */
  private proxyGenerate(id: number, req: GenerateRequest): Promise<void> {
    const ac = new AbortController();
    this.genControllers.set(id, ac);
    return apiGenerate(
      req,
      {
        onCtxPpl: (ppl) => this.post({ type: "gen", id, event: "ctx_ppl", data: { ppl } }),
        onPrefillPpl: (d) => this.post({ type: "gen", id, event: "prefill_ppl", data: d }),
        onUpdate: (u) => this.post({ type: "gen", id, event: "update", data: u }),
        onError: (msg) => this.post({ type: "gen", id, event: "error", data: { error: msg } }),
      },
      ac.signal
    ).finally(() => this.genControllers.delete(id));
  }

  /** 导出 Markdown：保存对话框 + 服务端导出接口 + 写文件。 */
  private async exportMd(id: string, title: string): Promise<{ path: string } | null> {
    const safe = (title || "document").replace(/[\\/:*?"<>|]/g, "_");
    const target = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(`${safe}.md`),
      filters: { Markdown: ["md"] },
    });
    if (!target) return null; // 用户取消
    const text = await fetchExportMd(id);
    await fs.promises.writeFile(target.fsPath, text, "utf-8");
    return { path: target.fsPath };
  }

  /** 选择本地 Markdown 文件 → {text, title}；取消返回 null。 */
  private async pickMdFile(): Promise<{ text: string; title: string } | null> {
    const picks = await vscode.window.showOpenDialog({
      canSelectMany: false,
      openLabel: "导入",
      filters: { Markdown: ["md", "markdown", "txt"] },
    });
    if (!picks || !picks.length) return null;
    const text = await fs.promises.readFile(picks[0].fsPath, "utf-8");
    return {
      text,
      title: path.basename(picks[0].fsPath).replace(/\.(md|markdown|txt)$/i, ""),
    };
  }

  // ------------------------------------------------------------- HTML 组装

  private makeHtml(webview: vscode.Webview, extUri: vscode.Uri): string {
    const js = webview.asWebviewUri(vscode.Uri.joinPath(extUri, "dist", "webview.js"));
    const css = webview.asWebviewUri(vscode.Uri.joinPath(extUri, "dist", "webview.css"));
    // 无 connect-src：webview 不直连网络，全部经 postMessage 代理
    const csp = [
      "default-src 'none'",
      // 'unsafe-inline' 仅样式：index.html 困惑度图例使用内联 style 属性
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src ${webview.cspSource}`,
      `font-src ${webview.cspSource}`,
      `img-src ${webview.cspSource} data:`,
    ].join("; ");
    return indexHtml
      .replace('href="style.css"', `href="${css}"`)
      .replace('<script type="module" src="app.js"></script>', `<script src="${js}"></script>`)
      .replace("</head>", `  <meta http-equiv="Content-Security-Policy" content="${csp}">\n</head>`);
  }
}

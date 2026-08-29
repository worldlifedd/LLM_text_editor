// 插件入口：命令注册、状态栏、装饰监听、面板接线。
import * as vscode from "vscode";
import { apiLoad, apiSkills, GenParams } from "./api";
import { Decorator, foldingRanges } from "./decorations";
import { newDocumentText, parseDoc } from "./docmodel";
import { GenerationController } from "./generation";
import { GtePanelProvider, PanelReady } from "./panel";
import { ensureServer, getStatus } from "./server";
import { StateStore } from "./state";

interface SavedState {
  params: GenParams;
  context_mode: string;
  skills: string[];
}

const DEFAULT_PARAMS: GenParams = {
  max_new_tokens: 256,
  do_sample: true,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
};

export function activate(context: vscode.ExtensionContext): void {
  const state = new StateStore();
  const decorator = new Decorator();

  // 状态栏
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.command = "gte.showPanel";
  statusBar.text = "GTE: 未连接";
  statusBar.show();

  function loadSaved(): SavedState {
    const saved = context.workspaceState.get<SavedState>("gte.params");
    if (saved) return saved;
    const cfg = vscode.workspace.getConfiguration("gte").get<GenParams & { context_mode?: string }>("params");
    return {
      params: cfg ? { ...DEFAULT_PARAMS, ...cfg } : DEFAULT_PARAMS,
      context_mode: cfg?.context_mode || "chat",
      skills: [],
    };
  }

  function refreshDecorations(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "markdown") return;
    const parsed = parseDoc(editor.document.getText());
    decorator.apply(editor, parsed, state.get(editor.document.uri.toString()));
  }

  let panel: GtePanelProvider | null = null;

  const gen = new GenerationController(state, decorator, {
    statusBar: (text, tooltip) => {
      statusBar.text = text;
      statusBar.tooltip = tooltip;
    },
    refreshDecorations,
    notifyPanel: (msg) => panel?.post(msg),
    getRequestState: () => {
      const s = loadSaved();
      return { params: s.params, context_mode: s.context_mode, skills: s.skills };
    },
  });

  // 把服务端状态反映到状态栏 + 面板。pollStatus 与 watchLoadResult 共用。
  function applyStatus(st: Awaited<ReturnType<typeof getStatus>>): void {
    // 生成进行中：保持醒目的工作状态提示（含停止入口），不被轮询覆盖
    if (gen.isGenerating) {
      statusBar.text = "$(sync~spin) GTE: 生成中…（再按 Ctrl+Enter 停止）";
      statusBar.tooltip = st.message;
      panel?.post({ type: "status", status: st });
      return;
    }
    const label = st.loading
      ? `$(sync~spin) GTE: 加载中…`
      : st.loaded
      ? `$(check) GTE: ${
          st.kind === "local" ? "本地" : st.kind === "llamacpp" ? "llama.cpp" : "API"
        }${st.generating ? "｜生成中…" : ""}`
      : "GTE: 未连接";
    statusBar.text = label;
    statusBar.tooltip = st.message;
    panel?.post({ type: "status", status: st });
  }

  async function pollStatus(): Promise<void> {
    try {
      applyStatus(await getStatus());
    } catch {
      statusBar.text = "GTE: 未连接";
    }
  }
  const timer = setInterval(() => void pollStatus(), 10000);

  // /api/load 是异步后台任务：立即返回 {accepted:true}，真正成败写进
  // status.message。仅靠 10s 轮询且不弹窗，加载失败时用户只看到状态栏默默变
  // "未连接"。此 watcher 在提交加载后快速轮询，loading 结束即主动弹成功/失败
  // 通知，并把加载期间状态实时推给状态栏与面板（不必等下一个 10s 周期）。
  let loadWatchAbort: AbortController | null = null;
  async function watchLoadResult(): Promise<void> {
    loadWatchAbort?.abort(); // 新一次加载提交，中止旧 watcher
    const ac = new AbortController();
    loadWatchAbort = ac;
    const deadline = Date.now() + 180000; // 大模型加载最长观察 3 分钟
    let sawLoading = false;
    while (!ac.signal.aborted && Date.now() < deadline) {
      const st = await getStatus().catch(async () => {
        await new Promise((r) => setTimeout(r, 1000));
        return null;
      });
      if (!st) continue;
      applyStatus(st);
      if (st.loading) sawLoading = true;
      if (sawLoading && !st.loading) {
        // loading 由 true→false：加载结束（成功或失败）
        if (st.loaded) {
          vscode.window.showInformationMessage(st.message || "✅ 模型加载成功");
        } else {
          vscode.window.showErrorMessage(st.message || "❌ 加载失败（未知原因）");
        }
        return;
      }
      await new Promise((r) => setTimeout(r, 800));
    }
  }

  // 侧边面板
  const provider = new GtePanelProvider(
    async (): Promise<PanelReady> => {
      await ensureServer(); // 面板打开即尝试连接/自动拉起
      const status = await getStatus();
      const skills = await apiSkills();
      const s = loadSaved();
      return { status, skills, params: s.params, context_mode: s.context_mode };
    },
    async (rawMsg: unknown) => {
      const m = rawMsg as Record<string, unknown>;
      try {
        await ensureServer();
      } catch (e) {
        vscode.window.showErrorMessage((e as Error).message);
        return;
      }
      const t = m.type as string;
      if (t === "load") {
        try {
          await apiLoad({
            mode: m.mode as string,
            model_path: m.model_path as string,
            base_url: m.base_url as string,
            api_key: m.api_key as string,
            model: m.model as string,
            n_gpu_layers: typeof m.n_gpu_layers === "number" ? m.n_gpu_layers : undefined,
            n_ctx: typeof m.n_ctx === "number" ? m.n_ctx : undefined,
          });
          vscode.window.showInformationMessage("已提交加载请求，正在后台加载…");
          void watchLoadResult(); // loading 结束后主动弹成功/失败通知
        } catch (e) {
          vscode.window.showErrorMessage(`加载失败：${(e as Error).message}`);
        }
      } else if (t === "generate") {
        const params = m.params as GenParams;
        const context_mode = (m.context_mode as string) || "chat";
        const skills = (m.skills as string[]) || [];
        await context.workspaceState.update("gte.params", { params, context_mode, skills });
        // 生成/停止同一按键切换
        if (gen.isGenerating) void gen.stop();
        else void gen.start();
      } else if (t === "stop") {
        void gen.stop();
      } else if (t === "refreshSkills") {
        try {
          const skills = await apiSkills();
          panel?.post({ type: "skills", skills });
        } catch (e) {
          vscode.window.showErrorMessage(`获取技能库失败：${(e as Error).message}`);
        }
      }
    }
  );
  panel = provider;

  // 命令
  context.subscriptions.push(
    // 生成/停止同一按键切换：用户随时知道 LLM 是否在工作
    vscode.commands.registerCommand("gte.generate", () =>
      gen.isGenerating ? void gen.stop() : void gen.start()
    ),
    vscode.commands.registerCommand("gte.stop", () => void gen.stop()),
    vscode.commands.registerCommand("gte.finalizeBlock", () => void gen.finalize()),
    vscode.commands.registerCommand("gte.addPromptBlock", () => void gen.addPrompt()),
    vscode.commands.registerCommand("gte.insertSkillBlock", () => void gen.insertSkillBlock()),
    vscode.commands.registerCommand("gte.toggleLock", () => gen.toggleLock()),
    vscode.commands.registerCommand("gte.newDocument", async () => {
      const doc = await vscode.workspace.openTextDocument({
        language: "markdown",
        content: newDocumentText(),
      });
      void vscode.window.showTextDocument(doc);
    }),
    vscode.commands.registerCommand("gte.showPanel", () =>
      void vscode.commands.executeCommand("gte.panel.focus")
    )
  );

  // 折叠提供器
  context.subscriptions.push(
    vscode.languages.registerFoldingRangeProvider(
      { language: "markdown" },
      {
        provideFoldingRanges(doc) {
          return foldingRanges(parseDoc(doc.getText()), doc);
        },
      }
    )
  );

  // 视图 / 监听
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(GtePanelProvider.viewType, provider),
    vscode.workspace.onDidChangeTextDocument((e) => gen.onDocChanged(e)),
    vscode.window.onDidChangeActiveTextEditor(() => gen.refresh()),
    vscode.workspace.onDidCloseTextDocument((d) => state.clear(d.uri.toString())),
    { dispose: () => clearInterval(timer) },
    decorator,
    statusBar
  );
}

export function deactivate(): void {
  /* 无后台线程需要清理 */
}

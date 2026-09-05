// 插件入口：命令注册 + 状态栏（服务/模型状态轮询）。
// Web 块编辑器（web/）以 WebviewPanel 编辑器标签页形态承载（webEditor.ts），
// 与浏览器版共用同一份前端代码（web/platform.js 平台抽象层）。
// 侧边栏（monitorPanel.ts）：后端参数管理 + 运行监控（显存/生成状态）。
import * as vscode from "vscode";
import { getStatus } from "./server";
import { WebEditorPanel } from "./webEditor";
import { MonitorPanel } from "./monitorPanel";

export function activate(context: vscode.ExtensionContext): void {
  // Web 编辑器（唯一功能入口）
  const openEditor = () => WebEditorPanel.createOrShow(context);
  context.subscriptions.push(
    vscode.commands.registerCommand("gte.openWebEditor", openEditor),
    // 旧命令别名：习惯「打开生成控制面板」的用户无感迁移
    vscode.commands.registerCommand("gte.showPanel", openEditor)
  );

  // 侧边栏：后端管理 + 监控（点击状态栏 GTE: 项聚焦）
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(MonitorPanel.viewType, new MonitorPanel(context))
  );

  // 状态栏：服务/模型状态（点击打开 Web 编辑器）
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.command = "gte.openWebEditor";
  statusBar.text = "GTE: 未连接";
  statusBar.tooltip = "GTE 生成式文本编辑器：点击打开 Web 编辑器";
  statusBar.show();

  async function pollStatus(): Promise<void> {
    try {
      const st = await getStatus();
      statusBar.text = st.loading
        ? "$(sync~spin) GTE: 加载中…"
        : st.loaded
        ? `$(check) GTE: ${
            st.kind === "local" ? "本地" : st.kind === "llamacpp" ? "llama.cpp" : "API"
          }${st.generating ? "｜生成中…" : ""}`
        : "GTE: 未连接";
      statusBar.tooltip = st.message || statusBar.tooltip;
    } catch {
      statusBar.text = "GTE: 未连接";
    }
  }
  void pollStatus();
  const timer = setInterval(() => void pollStatus(), 10000);

  context.subscriptions.push(
    { dispose: () => clearInterval(timer) },
    statusBar
  );
}

export function deactivate(): void {
  /* 无后台线程需要清理 */
}

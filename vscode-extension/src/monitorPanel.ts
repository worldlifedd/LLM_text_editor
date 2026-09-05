// 侧边栏面板：后端参数管理（切换后端类型/模型/上下文、加载/卸载）
// + 运行监控（显存/内存占用、生成状态与速率，轮询 /api/monitor）。
// Web 编辑器标签页负责写作本身；本面板承担「控制台」角色，两者共用同一
// 服务端，互为补充（面板加载模型后编辑器即可直接生成）。
import * as vscode from "vscode";
import {
  apiLoad, apiMonitor, apiStop, apiUnload, MonitorInfo,
} from "./api";
import { ensureServer } from "./server";

const FORM_KEY = "gte.monitor.form";

export interface MonitorForm {
  mode: "local" | "llamacpp" | "api";
  /** local：HuggingFace 名称 / 本地目录 */
  model_path: string;
  /** llamacpp：GGUF 文件路径（与 local 路径分开记忆，切换模式互不覆盖） */
  gguf_path: string;
  n_gpu_layers: number;
  n_ctx: number;
  base_url: string;
  api_key: string;
  model: string;
}

const DEFAULT_FORM: MonitorForm = {
  mode: "local",
  model_path: "",
  gguf_path: "",
  n_gpu_layers: -1,
  n_ctx: 4096,
  base_url: "",
  api_key: "",
  model: "",
};

function nonce(): string {
  let s = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

export class MonitorPanel implements vscode.WebviewViewProvider {
  public static readonly viewType = "gte.monitor";

  private view?: vscode.WebviewView;
  private timer?: vscode.Disposable;
  private loadWatchAbort: AbortController | null = null;

  constructor(private readonly context: vscode.ExtensionContext) {}

  public resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.makeHtml(view.webview.cspSource);
    view.webview.onDidReceiveMessage((m) => void this.handle(m));
    view.onDidChangeVisibility(() => {
      if (view.visible) {
        void this.pushMonitor();
      }
    });
    view.onDidDispose(() => {
      this.view = undefined;
    });
    // 初始表单 + 立即拉一次监控；此后 2s 轮询（面板可见时）
    void this.pushInit();
    const interval = setInterval(() => {
      if (this.view?.visible) void this.pushMonitor();
    }, 2000);
    this.timer = new vscode.Disposable(() => clearInterval(interval));
    this.context.subscriptions.push(this);
  }

  public dispose(): void {
    this.timer?.dispose();
    this.timer = undefined;
    this.loadWatchAbort?.abort();
  }

  private post(msg: unknown): void {
    void this.view?.webview.postMessage(msg);
  }

  private async pushInit(): Promise<void> {
    const form = {
      ...DEFAULT_FORM,
      ...(this.context.globalState.get<Partial<MonitorForm>>(FORM_KEY) || {}),
    };
    this.post({ type: "init", form });
    void this.pushMonitor();
  }

  private async pushMonitor(): Promise<void> {
    try {
      this.post({ type: "monitor", data: await apiMonitor() });
    } catch {
      this.post({ type: "offline" });
    }
  }

  // ------------------------------------------------------------- 消息处理

  private async handle(raw: unknown): Promise<void> {
    const m = raw as Record<string, unknown>;
    if (!m || typeof m !== "object") return;
    if (m.type === "form") {
      // 表单改动持久化（globalState，跨会话保留）
      await this.context.globalState.update(FORM_KEY, m.value ?? undefined);
      return;
    }
    if (m.type === "load") {
      await this.doLoad(m.value as MonitorForm);
    } else if (m.type === "unload") {
      try {
        await ensureServer();
        const r = await apiUnload();
        if (!r.unloaded) vscode.window.showInformationMessage("当前后端本就未加载模型");
        void this.pushMonitor();
      } catch (e) {
        vscode.window.showErrorMessage(`卸载失败：${(e as Error).message}`);
      }
    } else if (m.type === "stop") {
      try {
        await ensureServer();
        await apiStop();
      } catch (e) {
        vscode.window.showErrorMessage(`停止失败：${(e as Error).message}`);
      }
      void this.pushMonitor();
    }
  }

  private async doLoad(form: MonitorForm): Promise<void> {
    try {
      await ensureServer();
    } catch (e) {
      vscode.window.showErrorMessage((e as Error).message);
      return;
    }
    try {
      await apiLoad({
        mode: form.mode,
        // llamacpp 的模型路径存在 gguf_path（与 local 路径分开记忆）
        model_path: form.mode === "llamacpp" ? form.gguf_path : form.model_path,
        base_url: form.base_url,
        api_key: form.api_key,
        model: form.model,
        n_gpu_layers: form.n_gpu_layers,
        n_ctx: form.n_ctx,
      });
      vscode.window.showInformationMessage("已提交加载请求，正在后台加载…");
      void this.watchLoadResult();
    } catch (e) {
      vscode.window.showErrorMessage(`加载失败：${(e as Error).message}`);
    }
  }

  /** /api/load 是异步后台任务：loading 由 true→false 时主动弹成功/失败通知。 */
  private async watchLoadResult(): Promise<void> {
    this.loadWatchAbort?.abort(); // 新一次加载提交，中止旧 watcher
    const ac = new AbortController();
    this.loadWatchAbort = ac;
    const deadline = Date.now() + 180000; // 大模型加载最长观察 3 分钟
    let sawLoading = false;
    while (!ac.signal.aborted && Date.now() < deadline) {
      let st: MonitorInfo | null = null;
      try {
        st = await apiMonitor();
      } catch {
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }
      if (st.loading) sawLoading = true;
      if (sawLoading && !st.loading) {
        if (st.loaded) {
          vscode.window.showInformationMessage(st.message || "✅ 模型加载成功");
        } else {
          vscode.window.showErrorMessage(st.message || "❌ 加载失败（未知原因）");
        }
        void this.pushMonitor();
        return;
      }
      await new Promise((r) => setTimeout(r, 800));
    }
  }

  // ------------------------------------------------------------- HTML 组装

  private makeHtml(cspSource: string): string {
    const n = nonce();
    // 注意：内嵌脚本禁用模板字符串（外层是 TS 模板字面量，反引号需转义）
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="
  default-src 'none';
  style-src ${cspSource} 'unsafe-inline';
  script-src 'nonce-${n}';">
<style>
  body { padding: 0 12px 12px; font-size: 12px; }
  h2 {
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
    margin: 14px 0 6px; opacity: .8;
  }
  .row { margin: 6px 0; }
  label { display: block; margin-bottom: 2px; opacity: .8; }
  input, select {
    width: 100%; box-sizing: border-box; padding: 3px 6px;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px;
    font: inherit;
  }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .btns { display: flex; gap: 6px; margin-top: 8px; }
  button {
    flex: 1; padding: 4px 8px; font: inherit; cursor: pointer;
    background: var(--vscode-button-background); color: var(--vscode-button-foreground);
    border: none; border-radius: 2px;
  }
  button.secondary {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
  }
  .kv { display: flex; justify-content: space-between; gap: 8px; margin: 3px 0; }
  .kv .v { text-align: right; word-break: break-all; opacity: .9; }
  .bar {
    height: 8px; border-radius: 4px; margin: 4px 0 2px;
    background: var(--vscode-editorWidget-background, rgba(128,128,128,.2));
    overflow: hidden;
  }
  .bar > div { height: 100%; background: var(--vscode-charts-blue, #3794ff); transition: width .5s; }
  .muted { opacity: .6; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .ok { background: var(--vscode-testing-iconPassed, #73c991); }
  .warn { background: var(--vscode-charts-yellow, #cca700); }
  .spin { background: var(--vscode-charts-blue, #3794ff); }
  .off { background: var(--vscode-testing-iconFailed, #f14c4c); }
</style>
</head>
<body>
<h2>后端</h2>
<div class="row">
  <select id="mode">
    <option value="local">local（transformers）</option>
    <option value="llamacpp">llama.cpp（GGUF）</option>
    <option value="api">API（OpenAI 兼容）</option>
  </select>
</div>
<div id="f-local">
  <div class="row"><label>模型路径 / HuggingFace 名称</label>
    <input id="model_path" placeholder="Qwen/Qwen3.5-4B 或本地目录"></div>
</div>
<div id="f-llamacpp" hidden>
  <div class="row"><label>GGUF 模型路径</label>
    <input id="gguf_path" placeholder="path/to/model-Q4_K_M.gguf"></div>
  <div class="grid2">
    <div><label>GPU 层数（-1 全部 / 0 纯 CPU）</label><input id="n_gpu_layers" type="number" value="-1"></div>
    <div><label>上下文窗口</label><input id="n_ctx" type="number" value="4096"></div>
  </div>
</div>
<div id="f-api" hidden>
  <div class="row"><label>Base URL</label><input id="base_url" placeholder="http://127.0.0.1:8000/v1"></div>
  <div class="row"><label>API Key</label><input id="api_key" type="password"></div>
  <div class="row"><label>模型名</label><input id="model" placeholder="qwen3.5-4b"></div>
</div>
<div class="btns">
  <button id="btnLoad">加载 / 切换</button>
  <button id="btnUnload" class="secondary">卸载</button>
</div>

<h2>监控</h2>
<div class="row" id="state"><span class="muted">连接中…</span></div>
<div class="row muted" id="msg"></div>
<div id="modelInfo" hidden>
  <div class="kv"><span>模型</span><span class="v" id="mName"></span></div>
  <div class="kv"><span>设备</span><span class="v" id="mDevice"></span></div>
  <div class="kv"><span>量化</span><span class="v" id="mQuant"></span></div>
  <div class="kv"><span>上下文</span><span class="v" id="mCtx"></span></div>
</div>
<div id="gpuSection" hidden>
  <div class="kv"><span>显存</span><span class="v" id="gpuText"></span></div>
  <div class="bar"><div id="gpuBar" style="width:0"></div></div>
  <div class="kv muted"><span>torch 分配 / 预留</span><span class="v" id="torchText"></span></div>
</div>
<div class="kv" id="rssRow" hidden><span>服务进程内存</span><span class="v" id="rssText"></span></div>
<h2>生成</h2>
<div class="row" id="genState"><span class="muted">—</span></div>
<div class="kv"><span>token 数 / 用时</span><span class="v" id="genStats">—</span></div>
<div class="kv"><span>速率</span><span class="v" id="genTps">—</span></div>
<button id="btnStop" class="secondary" style="margin-top:8px; width:100%;">停止生成</button>

<script nonce="${n}">
(function () {
  "use strict";
  var vscode = acquireVsCodeApi();

  function $(id) { return document.getElementById(id); }

  // 按后端类型显隐对应字段区
  function applyMode() {
    var mode = $("mode").value;
    $("f-local").hidden = mode !== "local";
    $("f-llamacpp").hidden = mode !== "llamacpp";
    $("f-api").hidden = mode !== "api";
  }

  function readForm() {
    return {
      mode: $("mode").value,
      model_path: $("model_path").value.trim(),
      gguf_path: $("gguf_path").value.trim(),
      n_gpu_layers: parseInt($("n_gpu_layers").value, 10) || 0,
      n_ctx: parseInt($("n_ctx").value, 10) || 4096,
      base_url: $("base_url").value.trim(),
      api_key: $("api_key").value,
      model: $("model").value.trim(),
    };
  }

  function writeForm(f) {
    if (!f) return;
    $("mode").value = f.mode || "local";
    $("model_path").value = f.model_path || "";
    $("gguf_path").value = f.gguf_path || "";
    $("n_gpu_layers").value = String(f.n_gpu_layers != null ? f.n_gpu_layers : -1);
    $("n_ctx").value = String(f.n_ctx || 4096);
    $("base_url").value = f.base_url || "";
    $("api_key").value = f.api_key || "";
    $("model").value = f.model || "";
    applyMode();
  }

  var saveTimer = null;
  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      vscode.postMessage({ type: "form", value: readForm() });
    }, 500);
  }

  ["mode", "model_path", "gguf_path", "n_gpu_layers", "n_ctx", "base_url", "api_key", "model"]
    .forEach(function (id) {
      $(id).addEventListener("input", scheduleSave);
      $(id).addEventListener("change", scheduleSave);
    });
  $("mode").addEventListener("change", applyMode);

  $("btnLoad").addEventListener("click", function () {
    vscode.postMessage({ type: "load", value: readForm() });
  });
  $("btnUnload").addEventListener("click", function () {
    vscode.postMessage({ type: "unload" });
  });
  $("btnStop").addEventListener("click", function () {
    vscode.postMessage({ type: "stop" });
  });

  function fmtMb(mb) {
    if (mb == null) return "—";
    return mb >= 1024 ? (mb / 1024).toFixed(1) + " GB" : mb + " MB";
  }

  function renderMonitor(d) {
    // 状态行
    var dot, text;
    if (d.loading) { dot = "spin"; text = "加载中…"; }
    else if (d.generating) { dot = "spin"; text = "生成中…"; }
    else if (d.loaded) { dot = "ok"; text = "已就绪"; }
    else { dot = "warn"; text = "未加载模型"; }
    var kinds = { local: "transformers", llamacpp: "llama.cpp", api: "API" };
    $("state").innerHTML =
      '<span class="status-dot ' + dot + '"></span>' +
      (kinds[d.kind] || d.kind) + " · " + text;
    $("msg").textContent = d.message || "";

    // 模型信息
    var showModel = d.loaded && (d.model.name || d.model.base_url);
    $("modelInfo").hidden = !showModel;
    if (showModel) {
      $("mName").textContent = d.model.name || d.model.base_url || "—";
      $("mDevice").textContent = d.model.device || "—";
      $("mQuant").textContent = d.model.quant || "—";
      $("mCtx").textContent = d.model.context_size ? String(d.model.context_size) : "—";
    }

    // 显存（pynvml 缺失或无 GPU 时隐藏整节）
    var g = d.gpu || {};
    var hasGpu = g.total_mb != null && g.used_mb != null;
    $("gpuSection").hidden = !hasGpu;
    if (hasGpu) {
      $("gpuText").textContent = fmtMb(g.used_mb) + " / " + fmtMb(g.total_mb);
      $("gpuBar").style.width = Math.min(100, (g.used_mb / g.total_mb) * 100).toFixed(1) + "%";
      $("torchText").textContent =
        g.torch_allocated_mb != null
          ? fmtMb(g.torch_allocated_mb) + " / " + fmtMb(g.torch_reserved_mb)
          : "—";
    }

    // 进程内存
    $("rssRow").hidden = d.rss_mb == null;
    if (d.rss_mb != null) $("rssText").textContent = fmtMb(d.rss_mb);

    // 生成状态
    $("genState").innerHTML = d.generating
      ? '<span class="status-dot spin"></span>生成进行中'
      : '<span class="muted">空闲</span>';
    $("genStats").textContent = d.gen.tokens
      ? d.gen.tokens + " tok / " + (d.gen.elapsed_s != null ? d.gen.elapsed_s + " s" : "—")
      : "—";
    $("genTps").textContent = d.gen.tps != null ? d.gen.tps + " tok/s" : "—";
  }

  function renderOffline() {
    $("state").innerHTML = '<span class="status-dot off"></span>服务未连接';
    $("msg").textContent = "";
  }

  window.addEventListener("message", function (ev) {
    var m = ev.data;
    if (!m || typeof m !== "object") return;
    if (m.type === "init") writeForm(m.form);
    else if (m.type === "monitor") renderMonitor(m.data);
    else if (m.type === "offline") renderOffline();
  });
})();
</script>
</body>
</html>`;
  }
}

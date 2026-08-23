// 侧边 WebviewView 面板：后端连接、生成参数、技能库、困惑度指标。
// Webview 不直连网络，所有 HTTP 经扩展主进程代理（postMessage 往返）。
import * as vscode from "vscode";
import { GenParams } from "./api";
import { serverUrl } from "./server";

export interface PanelReady {
  status: { kind: string; loaded: boolean; loading: boolean; message: string; generating: boolean };
  skills: { name: string; description: string }[];
  params: GenParams;
  context_mode: string;
}

export class GtePanelProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "gte.panel";
  private view?: vscode.WebviewView;
  private lastStatus?: string;
  private lastPpl?: string;

  constructor(
    private resolveReady: () => Promise<PanelReady>,
    private handleMessage: (msg: unknown) => Promise<void> | void
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.html();

    webviewView.webview.onDidReceiveMessage((msg) => {
      void this.handleMessage(msg);
    });

    void this.refresh();
  }

  async refresh(): Promise<void> {
    if (!this.view) return;
    try {
      const ready = await this.resolveReady();
      this.post({ type: "status", status: ready.status });
      this.post({ type: "skills", skills: ready.skills });
      this.post({ type: "params", params: ready.params, context_mode: ready.context_mode });
    } catch (e) {
      this.post({ type: "status", status: { message: `❌ ${(e as Error).message}` } });
    }
  }

  post(msg: unknown): void {
    if (this.view) void this.view.webview.postMessage(msg);
  }

  private html(): string {
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<style>
:root { color-scheme: light dark; }
body { font-family: var(--vscode-font-family); font-size: 13px; padding: 0 12px 20px; }
h3 { margin: 14px 0 6px; font-size: 13px; }
.section { border: 1px solid var(--vscode-panel-border); border-radius: 4px; padding: 8px 10px; margin-bottom: 10px; }
.row { display: flex; gap: 6px; align-items: center; margin: 4px 0; }
.row label { width: 96px; flex: none; color: var(--vscode-descriptionForeground); }
input[type=text], input[type=password], select, input[type=number] {
  flex: 1; min-width: 0; background: var(--vscode-input-background);
  color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border);
  padding: 3px 6px; border-radius: 2px;
}
input[type=checkbox] { vertical-align: middle; }
button {
  background: var(--vscode-button-background); color: var(--vscode-button-foreground);
  border: none; padding: 4px 10px; border-radius: 2px; cursor: pointer; margin-top: 6px;
}
button:hover { background: var(--vscode-button-hoverBackground); }
button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
.status { margin-top: 6px; font-size: 12px; color: var(--vscode-descriptionForeground); white-space: pre-wrap; word-break: break-all; }
#skillsList label { display: block; width: auto; margin: 2px 0; }
.metric { font-size: 13px; margin: 3px 0; }
.metric b { font-size: 14px; }
.legend span { padding: 1px 6px; border-radius: 3px; margin-right: 6px; font-size: 12px; }
svg { width: 100%; height: 130px; }
.tip { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
</style>
</head>
<body>
  <h3>🔌 后端</h3>
  <div class="section">
    <div class="row"><label>模式</label>
      <select id="mode">
        <option value="local">本地模型（transformers）</option>
        <option value="llamacpp">llama.cpp（GGUF 量化）</option>
        <option value="api">OpenAI 兼容 API</option>
      </select>
    </div>
    <div id="localRow" class="row"><label>模型路径/ID</label>
      <input type="text" id="modelPath" value="Qwen/Qwen2.5-0.5B-Instruct"></div>
    <div id="llamaRow" class="row" style="display:none"><label>GGUF 路径/仓库</label>
      <input type="text" id="llamaPath" placeholder="models/qwen2.5-0.5b-instruct-q4_k_m.gguf"></div>
    <div id="llamaRow2" class="row" style="display:none"><label>n_gpu_layers</label>
      <input type="number" id="llamaGpu" value="-1" min="-1" max="100" step="1"></div>
    <div id="llamaRow3" class="row" style="display:none"><label>n_ctx</label>
      <input type="number" id="llamaCtx" value="4096" min="512" max="32768" step="512"></div>
    <div id="apiRow" class="row" style="display:none"><label>base_url</label>
      <input type="text" id="apiBase" value="https://api.openai.com/v1"></div>
    <div id="apiRow2" class="row" style="display:none"><label>API Key</label>
      <input type="password" id="apiKey"></div>
    <div id="apiRow3" class="row" style="display:none"><label>模型名</label>
      <input type="text" id="apiModel" value="gpt-4o-mini"></div>
    <div id="llamaTip" class="tip" style="display:none">
      本地 .gguf 文件路径或 HF GGUF 仓库 ID（自动下载）。n_gpu_layers：-1=全部层上 GPU，0=纯 CPU。
    </div>
    <button id="loadBtn">加载 / 连接</button>
    <div class="status" id="status">未加载</div>
  </div>

  <h3>🎛 生成参数</h3>
  <div class="section">
    <div class="row"><label>max_new_tokens</label><input type="number" id="pMax" value="256" min="16" step="16"></div>
    <div class="row"><label>do_sample</label><input type="checkbox" id="pSample" checked></div>
    <div class="row"><label>temperature</label><input type="number" id="pTemp" value="0.8" min="0.1" max="2" step="0.05"></div>
    <div class="row"><label>top_k</label><input type="number" id="pTopK" value="50" min="1" max="200" step="1"></div>
    <div class="row"><label>top_p</label><input type="number" id="pTopP" value="0.95" min="0.05" max="1" step="0.05"></div>
    <div class="row"><label>rep_penalty</label><input type="number" id="pRep" value="1.1" min="1" max="2" step="0.05"></div>
    <div class="row"><label>上下文模式</label>
      <select id="pMode">
        <option value="chat">chat（聊天模板）</option>
        <option value="prefix">prefix（【指令】/【正文】）</option>
        <option value="raw">raw（原文裸拼接）</option>
      </select>
    </div>
    <button id="genBtn">▶ 生成（当前文档）</button>
    <button class="secondary" id="stopBtn">⏹ 停止</button>
  </div>

  <h3>🧩 技能库</h3>
  <div class="section">
    <div id="skillsList"><span class="tip">技能库为空</span></div>
    <button class="secondary" id="refreshSkills">🔄 刷新技能库</button>
  </div>

  <h3>📊 困惑度指标</h3>
  <div class="section">
    <div class="metric">上下文困惑度：<b id="ctxPpl">—</b></div>
    <div class="metric">平均生成困惑度：<b id="avgPpl">—</b></div>
    <div class="metric" id="cacheInfo" style="font-size:12px"></div>
    <div class="legend">
      <span style="background:#86E294">ppl≈1</span>
      <span style="background:#FFE282">ppl≈20</span>
      <span style="background:#FF766C">ppl≥200</span>
      <span style="background:rgba(160,160,160,0.5)">灰=手动编辑</span>
    </div>
    <svg id="chart" viewBox="0 0 520 130" preserveAspectRatio="none"></svg>
    <div class="tip">绿=模型确定 → 红=模型困惑（全部生成块累计）</div>
  </div>

<script>
const vscode = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);

function setMode(m) {
  const api = m === "api";
  const llama = m === "llamacpp";
  $("localRow").style.display = api || llama ? "none" : "";
  $("llamaRow").style.display = llama ? "" : "none";
  $("llamaRow2").style.display = llama ? "" : "none";
  $("llamaRow3").style.display = llama ? "" : "none";
  $("llamaTip").style.display = llama ? "" : "none";
  $("apiRow").style.display = api ? "" : "none";
  $("apiRow2").style.display = api ? "" : "none";
  $("apiRow3").style.display = api ? "" : "none";
}

function collectParams() {
  return {
    max_new_tokens: parseInt($("pMax").value, 10) || 256,
    do_sample: $("pSample").checked,
    temperature: parseFloat($("pTemp").value) || 0.8,
    top_k: parseInt($("pTopK").value, 10) || 50,
    top_p: parseFloat($("pTopP").value) || 0.95,
    repetition_penalty: parseFloat($("pRep").value) || 1.1,
  };
}
function collectSkills() {
  return Array.from(document.querySelectorAll("#skillsList input:checked")).map((c) => c.value);
}

$("mode").addEventListener("change", (e) => setMode(e.target.value));
$("loadBtn").addEventListener("click", () => {
  const mode = $("mode").value;
  if (mode === "local") {
    vscode.postMessage({ type: "load", mode, model_path: $("modelPath").value });
  } else if (mode === "llamacpp") {
    vscode.postMessage({
      type: "load", mode, model_path: $("llamaPath").value,
      n_gpu_layers: parseInt($("llamaGpu").value, 10) || 0,
      n_ctx: parseInt($("llamaCtx").value, 10) || 4096,
    });
  } else {
    vscode.postMessage({
      type: "load", mode, base_url: $("apiBase").value,
      api_key: $("apiKey").value, model: $("apiModel").value,
    });
  }
});
$("genBtn").addEventListener("click", () => {
  vscode.postMessage({
    type: "generate", params: collectParams(), context_mode: $("pMode").value,
    skills: collectSkills(),
  });
});
$("stopBtn").addEventListener("click", () => vscode.postMessage({ type: "stop" }));
$("refreshSkills").addEventListener("click", () => vscode.postMessage({ type: "refreshSkills" }));

function drawChart(series) {
  const svg = $("chart");
  if (!series || !series.length) {
    svg.innerHTML = '<text x="10" y="20" fill="#888" font-size="12">暂无生成数据</text>';
    return;
  }
  const W = 520, H = 130, pad = 8;
  const n = series.length;
  const log = series.map((p) => Math.log(Math.max(p, 1)));
  const logMax = Math.max(...log, Math.log(2));
  const xAt = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
  const yAt = (v) => H - pad - (v / logMax) * (H - 2 * pad);
  const pts = series.map((p, i) => xAt(i) + "," + yAt(Math.log(Math.max(p, 1)))).join(" ");
  const win = 10;
  const ma = series.map((_, i) => {
    const lo = Math.max(0, i + 1 - win);
    const seg = series.slice(lo, i + 1);
    return Math.exp(seg.reduce((a, p) => a + Math.log(Math.max(p, 1e-9)), 0) / seg.length);
  });
  const maPts = ma.map((p, i) => xAt(i) + "," + yAt(Math.log(Math.max(p, 1)))).join(" ");
  svg.innerHTML =
    '<polyline fill="none" stroke="#888" stroke-width="1" points="' + pts + '"/>' +
    '<polyline fill="none" stroke="#4FC1FF" stroke-width="2" points="' + maPts + '"/>';
}

window.addEventListener("message", (ev) => {
  const msg = ev.data;
  if (msg.type === "status") {
    const s = msg.status;
    $("status").textContent = s.message || "未加载";
    if (s.kind) setMode(s.kind);
  } else if (msg.type === "skills") {
    const box = $("skillsList");
    if (!msg.skills || !msg.skills.length) {
      box.innerHTML = '<span class="tip">技能库为空</span>';
    } else {
      box.innerHTML = msg.skills
        .map((s) => '<label><input type="checkbox" value="' + s.name + '"> ' + s.name +
          (s.description ? ' <span style="color:#888">— ' + s.description + "</span>" : "") + "</label>")
        .join("");
    }
  } else if (msg.type === "params") {
    $("pMax").value = msg.params.max_new_tokens;
    $("pSample").checked = !!msg.params.do_sample;
    $("pTemp").value = msg.params.temperature;
    $("pTopK").value = msg.params.top_k;
    $("pTopP").value = msg.params.top_p;
    $("pRep").value = msg.params.repetition_penalty;
    $("pMode").value = msg.context_mode || "chat";
  } else if (msg.type === "ppl") {
    $("ctxPpl").textContent = msg.ctxPpl == null ? "—（本地/llama.cpp 模式）" : String(msg.ctxPpl);
    $("avgPpl").textContent = msg.avgPpl == null ? "—" : msg.avgPpl.toFixed(2);
    $("cacheInfo").textContent = msg.cacheInfo || "";
    drawChart(msg.series);
  } else if (msg.type === "gen") {
    $("genBtn").disabled = msg.generating;
  }
});
</script>
</body>
</html>`;
  }
}

// 无头服务生命周期：探活、自动拉起 server.py、轮询就绪。
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { spawn, ChildProcess } from "child_process";

export interface ServerStatus {
  kind: string;
  loaded: boolean;
  loading: boolean;
  message: string;
  generating: boolean;
  /** 服务进程解释器与 PID（server.py 上报，旧版服务可能缺省） */
  python?: string;
  pid?: number;
  /** 思维链能力：supported=yes/no/unknown；toggleable=可开关 */
  reasoning?: { supported?: string; toggleable?: boolean };
}

let child: ChildProcess | null = null;
let output: vscode.OutputChannel | null = null;

/** 写入「GTE Server」输出面板（其余模块复用：如加载失败详情）。 */
export function log(msg: string): void {
  if (!output) output = vscode.window.createOutputChannel("GTE Server");
  output.appendLine(msg);
}

export function serverUrl(): string {
  const cfg = vscode.workspace.getConfiguration("gte");
  return (cfg.get<string>("serverUrl") || "http://127.0.0.1:8907").replace(/\/+$/, "");
}

export function serverPort(): number {
  const m = /:(\d+)\/?$/.exec(serverUrl());
  return m ? parseInt(m[1], 10) : 8907;
}

function findServerScript(): string | null {
  const cfg = vscode.workspace.getConfiguration("gte");
  const explicit = cfg.get<string>("serverScript");
  if (explicit) return explicit;
  // 扩展根目录：打包后 __dirname 为 <ext>/dist，其上级即扩展根（含 python/）
  // 注意不能再用 dirname(dirname(...))——那会得到 .vscode/extensions（上级的上级）
  const extRoot = path.dirname(__dirname);
  const candidates = [
    path.join(extRoot, "python", "server.py"),
    path.join(extRoot, "server.py"),
    // 开发态（F5，esbuild 输出到 vscode-extension/dist 时）仓库根在扩展目录上级
    path.join(extRoot, "..", "server.py"),
    path.join(extRoot, "src", "server.py"),
  ];
  // 工作区文件夹
  const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (ws) candidates.unshift(path.join(ws, "server.py"));
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch {
      /* ignore */
    }
  }
  return null;
}

async function waitUntilReady(timeoutMs = 60000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(2000) });
      if (r.ok) return true;
    } catch {
      /* not ready yet */
    }
    await new Promise((res) => setTimeout(res, 500));
  }
  return false;
}

/** 探活并取回 status；不可达/非 200 时返回 null。 */
async function fetchStatus(url: string): Promise<ServerStatus | null> {
  try {
    const r = await fetch(`${url}/api/status`, { signal: AbortSignal.timeout(2000) });
    if (!r.ok) return null;
    return (await r.json().catch(() => null)) as ServerStatus | null;
  } catch {
    return null;
  }
}

/** 路径归一化（分隔符/大小写/尾斜杠），用于跨平台比较 python 路径。 */
function normPath(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function isAbsPath(p: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(p) || /^[\\/]/.test(p);
}

/** 等端口释放（旧进程退出需一小段时间）。 */
async function waitPortFree(url: string, timeoutMs = 5000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!(await fetchStatus(url))) return true;
    await new Promise((res) => setTimeout(res, 300));
  }
  return false;
}

/** 确保服务可达；不可达且允许时自动拉起。返回服务地址。 */
export async function ensureServer(): Promise<string> {
  const url = serverUrl();
  const cfg = vscode.workspace.getConfiguration("gte");
  const py = cfg.get<string>("pythonCommand") || "python";
  log(`ensureServer: gte.pythonCommand = "${py}"`);
  log(`ensureServer: 探活 ${url}/api/status ...`);
  const st = await fetchStatus(url);
  if (st) {
    // 已有服务在跑。校验其解释器是否与 gte.pythonCommand 一致：不一致
    // 说明是改配置前残留的旧进程（常见于换 conda 环境后），继续复用会在
    // 加载模型时报“llama-cpp-python 未安装”等环境错误 → 自动结束重启。
    if (st.python && st.pid && isAbsPath(py) && normPath(st.python) !== normPath(py)) {
      log(
        `ensureServer: 端口服务由 ${st.python}（pid=${st.pid}）运行，` +
          `与配置 gte.pythonCommand=${py} 不一致，自动重启为配置解释器`
      );
      try {
        process.kill(st.pid);
      } catch (e) {
        const msg =
          `端口服务由 ${st.python}（pid=${st.pid}）运行，与 gte.pythonCommand` +
          `（${py}）不一致且无法自动结束（${(e as Error).message}），请手动结束后重试`;
        log(`ensureServer: ${msg}`);
        throw new Error(msg);
      }
      if (!(await waitPortFree(url))) {
        throw new Error(`旧服务（pid=${st.pid}）结束超时，请稍后重试`);
      }
    } else {
      log(`ensureServer: 服务已可达（python: ${st.python || "未知"}）`);
      return url;
    }
  } else {
    log(`ensureServer: 探活失败，将尝试拉起`);
  }

  if (!cfg.get<boolean>("autoStartServer")) {
    throw new Error(
      `服务不可达（${url}），且 gte.autoStartServer 已关闭。请先运行：python server.py`
    );
  }
  if (child && !child.killed) {
    throw new Error(`服务启动中…（${url}）`);
  }

  const script = findServerScript();
  if (!script) {
    const msg = `未找到 server.py：请在设置 gte.serverScript 中指定，或把 server.py 放入工作区/扩展目录`;
    log(`ensureServer: ${msg}`);
    throw new Error(msg);
  }
  log(`ensureServer: 启动服务 ${py} ${script} --port ${serverPort()}`);
  child = spawn(py, [script, "--port", String(serverPort())], {
    cwd: path.dirname(script),
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.once("error", (e) => log(`server.py 启动失败：${e.message}（检查 gte.pythonCommand）`));
  child.stdout?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.stderr?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.on("exit", (code) => {
    log(`server.py 退出，code=${code}`);
    child = null;
  });

  if (await waitUntilReady()) {
    log(`ensureServer: 服务就绪`);
    return url;
  }
  throw new Error(`服务启动超时（${url}）。查看「GTE Server」输出面板了解详情`);
}

export function getStatus(): Promise<ServerStatus> {
  return fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(3000) }).then(
    (r) => r.json() as Promise<ServerStatus>
  );
}

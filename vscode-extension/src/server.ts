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
}

let child: ChildProcess | null = null;
let output: vscode.OutputChannel | null = null;

function log(msg: string): void {
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

/** 确保服务可达；不可达且允许时自动拉起。返回服务地址。 */
export async function ensureServer(): Promise<string> {
  const url = serverUrl();
  try {
    const r = await fetch(`${url}/api/status`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) return url;
  } catch {
    /* fall through to spawn */
  }

  const cfg = vscode.workspace.getConfiguration("gte");
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
    throw new Error(
      `未找到 server.py：请在设置 gte.serverScript 中指定，或把 server.py 放入工作区/扩展目录`
    );
  }
  const py = cfg.get<string>("pythonCommand") || "python";
  log(`启动服务：${py} ${script} --port ${serverPort()}`);
  child = spawn(py, [script, "--port", String(serverPort())], {
    cwd: path.dirname(script),
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.stdout?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.stderr?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.on("exit", (code) => {
    log(`server.py 退出，code=${code}`);
    child = null;
  });

  if (await waitUntilReady()) return url;
  throw new Error(`服务启动超时（${url}）。查看「GTE Server」输出面板了解详情`);
}

export function getStatus(): Promise<ServerStatus> {
  return fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(3000) }).then(
    (r) => r.json() as Promise<ServerStatus>
  );
}

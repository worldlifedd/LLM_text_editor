// 服务端 HTTP 客户端 + SSE 流解析。
import { serverUrl } from "./server";

export interface GenParams {
  max_new_tokens: number;
  do_sample: boolean;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
}

export interface GenerateRequest {
  blocks: { type: "prompt" | "generate"; content: string }[];
  active_text: string;
  skills: string[];
  params: GenParams;
  context_mode: string;
}

export interface GenUpdate {
  cum_text: string;
  token_texts: string[];
  token_ppls: (number | null)[];
  final: boolean;
  cache_info?: string;
}

export interface SkillInfo {
  name: string;
  description: string;
}

export interface GenerateHandlers {
  onCtxPpl?: (ppl: number) => void;
  onUpdate: (u: GenUpdate) => void;
  onError: (msg: string) => void;
  onDone?: () => void;
}

async function postJson<T>(api: string, body: unknown, timeoutMs = 8000): Promise<T> {
  const r = await fetch(`${serverUrl()}${api}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { error?: string };
      if (j && j.error) msg = j.error;
    } catch {
      /* keep default */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export function apiStatus(): Promise<{ kind: string; loaded: boolean; loading: boolean; message: string; generating: boolean }> {
  return fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(3000) }).then(
    (r) => r.json() as Promise<{ kind: string; loaded: boolean; loading: boolean; message: string; generating: boolean }>
  );
}

export function apiLoad(req: {
  mode: string;
  model_path?: string;
  base_url?: string;
  api_key?: string;
  model?: string;
  n_gpu_layers?: number; // llamacpp：GPU offload 层数（-1=全部，0=纯 CPU）
  n_ctx?: number;        // llamacpp：上下文窗口
}): Promise<{ accepted: boolean }> {
  return postJson("/api/load", req);
}

export function apiStop(): Promise<{ stopped: boolean }> {
  return postJson("/api/stop", {});
}

export function apiSkills(): Promise<SkillInfo[]> {
  return postJson("/api/skills", {});
}

/** POST /api/generate 并解析 SSE 流。signal 可中止（停止）。 */
export async function apiGenerate(
  req: GenerateRequest,
  handlers: GenerateHandlers,
  signal: AbortSignal
): Promise<void> {
  const r = await fetch(`${serverUrl()}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { error?: string };
      if (j && j.error) msg = j.error;
    } catch {
      /* keep default */
    }
    handlers.onError(msg);
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let eventName = "";

  const handleData = (dataStr: string) => {
    let data: unknown;
    try {
      data = JSON.parse(dataStr);
    } catch {
      return;
    }
    if (eventName === "ctx_ppl") {
      handlers.onCtxPpl?.((data as { ppl: number }).ppl);
    } else if (eventName === "update") {
      const u = data as GenUpdate;
      handlers.onUpdate(u);
      if (u.final) handlers.onDone?.();
    } else if (eventName === "error") {
      handlers.onError((data as { error: string }).error || "生成失败");
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, "");
      buf = buf.slice(idx + 1);
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        handleData(line.slice(5).trim());
        eventName = "";
      }
    }
  }
}

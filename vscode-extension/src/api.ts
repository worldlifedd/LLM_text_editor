// 服务端 HTTP 客户端 + SSE 流解析。
import { serverUrl } from "./server";

export interface GenParams {
  max_new_tokens: number;
  do_sample: boolean;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  /** 思考模式（推理模型）：null=模型默认；false=关闭；true=强制开启 */
  enable_thinking?: boolean | null;
}

export interface GenerateRequest {
  blocks: { type: "prompt" | "system" | "cot" | "generate"; content: string }[];
  active_text: string;
  skills: string[];
  params: GenParams;
  context_mode: string;
}

export interface GenUpdate {
  cum_text: string;
  /** 累计思维链（推理模型；非推理模型为空） */
  reasoning_cum?: string;
  token_texts: string[];
  token_ppls: (number | null)[];
  /** 思维链困惑度（本地/llama.cpp 后端；API 无 reasoning logprobs） */
  reasoning_token_texts?: string[];
  reasoning_token_ppls?: (number | null)[];
  /** 思考区是否闭合（模型输出过闭标签）；未闭合时插件端不补写闭标签，
   *  cot 保持"续写思考"状态（所见即所得，用户可删/留闭标签控制行为） */
  reasoning_closed?: boolean;
  final: boolean;
  cache_info?: string;
}

export interface SkillInfo {
  name: string;
  description: string;
  /** 完整指令体（供固化为文档内 system 块） */
  instructions: string;
}

export interface PrefillPpl {
  /** 活动块文本（含手动编辑部分）的逐 token 困惑度片段；
   * join(token_texts) 为 active_text 的后缀 */
  token_texts: string[];
  token_ppls: (number | null)[];
}

export interface GenerateHandlers {
  onCtxPpl?: (ppl: number) => void;
  onPrefillPpl?: (d: PrefillPpl) => void;
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
  // 服务端为 GET 端点（POST 会返回 405）
  return fetch(`${serverUrl()}/api/skills`, { signal: AbortSignal.timeout(3000) }).then(
    (r) => r.json() as Promise<SkillInfo[]>
  );
}

// ---------------------------------------------------------------- 文档 CRUD
// 与 web/api.js 的文档端点一一对应（webview 经 postMessage 代理复用本层）。

export interface DocMeta {
  id: string;
  title: string;
  updated_at: string;
}

export interface DocBlock {
  type: "prompt" | "system" | "cot" | "generate";
  content: string;
  ppl?: { token_texts: string[]; token_ppls: (number | null)[] } | null;
  closed?: boolean | null;
}

export interface Doc {
  id: string;
  title: string;
  updated_at?: string;
  blocks: DocBlock[];
}

export function apiListDocs(): Promise<DocMeta[]> {
  return fetch(`${serverUrl()}/api/docs`, { signal: AbortSignal.timeout(3000) }).then(
    (r) => r.json() as Promise<DocMeta[]>
  );
}

export function apiGetDoc(id: string): Promise<Doc> {
  return fetch(`${serverUrl()}/api/docs/${encodeURIComponent(id)}`, {
    signal: AbortSignal.timeout(3000),
  }).then((r) => r.json() as Promise<Doc>);
}

export function apiSaveDoc(doc: {
  id: string;
  title: string;
  blocks: DocBlock[];
}): Promise<{ id: string }> {
  return postJson("/api/docs", doc);
}

export function apiDeleteDoc(id: string): Promise<{ deleted: boolean }> {
  return fetch(`${serverUrl()}/api/docs/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(3000),
  }).then((r) => r.json() as Promise<{ deleted: boolean }>);
}

export function apiImportMarkdown(
  text: string,
  title: string
): Promise<{ id: string; title: string; blocks: DocBlock[] }> {
  return postJson("/api/docs/import_md", { text, title });
}

/** 拉取导出的 Markdown 文本（主进程代存文件）。 */
export async function fetchExportMd(id: string): Promise<string> {
  const r = await fetch(`${serverUrl()}/api/docs/${encodeURIComponent(id)}/export_md`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.text();
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
    } else if (eventName === "prefill_ppl") {
      handlers.onPrefillPpl?.(data as PrefillPpl);
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

// REST + SSE 客户端（可移植内核：传输层仅用 fetch/ReadableStream/TextDecoder，
// VSCode Webview 移植时替换本文件为 postMessage 代理即可）。

async function jsonOrThrow(res) {
  if (res.ok) return res.json();
  let msg = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    if (body && body.error) msg = body.error;
  } catch { /* 保留 HTTP 状态码文案 */ }
  throw new Error(msg);
}

export async function apiStatus() {
  return jsonOrThrow(await fetch("/api/status"));
}

export async function apiLoad(body) {
  return jsonOrThrow(await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }));
}

export async function apiSkills() {
  return jsonOrThrow(await fetch("/api/skills"));
}

export async function apiStop() {
  return jsonOrThrow(await fetch("/api/stop", { method: "POST" }));
}

// ---------------------------------------------------------------- 文档 CRUD

export async function apiListDocs() {
  return jsonOrThrow(await fetch("/api/docs"));
}

export async function apiGetDoc(id) {
  return jsonOrThrow(await fetch(`/api/docs/${encodeURIComponent(id)}`));
}

export async function apiSaveDoc(doc) {
  return jsonOrThrow(await fetch("/api/docs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(doc),
  }));
}

export async function apiDeleteDoc(id) {
  return jsonOrThrow(
    await fetch(`/api/docs/${encodeURIComponent(id)}`, { method: "DELETE" })
  );
}

export async function apiImportMarkdown(text, title) {
  return jsonOrThrow(await fetch("/api/docs/import_md", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, title }),
  }));
}

/** 导出 Markdown（浏览器下载）。 */
export function exportMarkdownUrl(id) {
  return `/api/docs/${encodeURIComponent(id)}/export_md`;
}

// ---------------------------------------------------------------- SSE 生成流

/**
 * POST /api/generate 并解析 SSE 流。
 * 事件：ctx_ppl {ppl} / prefill_ppl {token_texts, token_ppls} /
 * update {cum_text, token_texts, token_ppls, reasoning_cum,
 * reasoning_token_texts, reasoning_token_ppls,
 * reasoning_closed, final, cache_info} / error {error}。
 * 注意：cum_text / token_texts 仅为本轮新生成文本（不含活动块前缀），
 * join(token_texts) === cum_text 恒成立；prefill_ppl 的 token_texts
 * 是活动块前缀的后缀（join 结果为 active_text 的后缀），为 prefill
 * 阶段对活动块文本（含手动编辑部分）的逐 token 打分。
 */
export async function apiGenerate(req, cb, signal) {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.error) msg = body.error;
    } catch { /* 保留状态码文案 */ }
    throw new Error(msg);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const rawEvent = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseSseEvent(rawEvent);
      if (!ev) continue;
      if (ev.event === "update") cb.onUpdate(ev.data);
      else if (ev.event === "ctx_ppl") cb.onCtxPpl(ev.data.ppl);
      else if (ev.event === "prefill_ppl") {
        if (cb.onPrefillPpl) cb.onPrefillPpl(ev.data);
      } else if (ev.event === "error") {
        if (cb.onError) cb.onError(ev.data.error);
      }
    }
  }
}

function parseSseEvent(raw) {
  let event = null;
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (event === null || !data) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return null;
  }
}

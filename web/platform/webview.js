// VSCode Webview 平台实现：postMessage RPC 代理。
// webview 不直连网络——所有 HTTP（含 SSE 生成流）经扩展主进程代理往返：
//   webview → 主进程：{type:"req", id, method, args} / {type:"abort", id}
//   主进程 → webview：{type:"resp", id, ok, data|error}
//                      {type:"gen", id, event, data}（generate 的 SSE 事件逐个转发）

let _api = null;
/** acquireVsCodeApi() 只能调用一次，缓存实例。 */
function vs() {
  if (!_api) _api = acquireVsCodeApi();
  return _api;
}

let nextId = 1;
const pending = new Map(); // id -> {resolve, reject}
const genCbs = new Map();  // id -> 生成事件回调集（generate 流式请求专用）

window.addEventListener("message", (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== "object") return;
  if (msg.type === "resp") {
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    genCbs.delete(msg.id);
    if (msg.ok) p.resolve(msg.data);
    else p.reject(new Error(msg.error || "请求失败"));
  } else if (msg.type === "gen") {
    const cb = genCbs.get(msg.id);
    if (!cb) return;
    if (msg.event === "update") {
      if (cb.onUpdate) cb.onUpdate(msg.data);
    } else if (msg.event === "ctx_ppl") {
      if (cb.onCtxPpl) cb.onCtxPpl(msg.data.ppl);
    } else if (msg.event === "prefill_ppl") {
      if (cb.onPrefillPpl) cb.onPrefillPpl(msg.data);
    } else if (msg.event === "error") {
      if (cb.onError) cb.onError(msg.data.error);
    }
  }
});

function call(method, args) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    vs().postMessage({ type: "req", id, method, args: args || {} });
  });
}

export const platform = {
  // ---------------------------------------------------------------- 传输
  apiStatus: () => call("status"),
  apiLoad: (body) => call("load", body),
  apiSkills: () => call("skills"),
  apiStop: () => call("stop"),
  apiListDocs: () => call("docs.list"),
  apiGetDoc: (id) => call("docs.get", { id }),
  apiSaveDoc: (doc) => call("docs.save", doc),
  apiDeleteDoc: (id) => call("docs.delete", { id }),
  apiImportMarkdown: (text, title) => call("docs.importMd", { text, title }),

  /** SSE 生成流：req 发出后立即返回，事件由 {type:"gen"} 消息驱动、resp 结束。 */
  apiGenerate(req, cb, signal) {
    const id = nextId++;
    genCbs.set(id, cb);
    const onAbort = () => {
      genCbs.delete(id);
      const p = pending.get(id);
      if (p) {
        pending.delete(id);
        p.resolve(); // 本地立即视为流结束（app.js 以 signal.aborted 区分）
      }
      vs().postMessage({ type: "abort", id });
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return Promise.resolve();
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }
    return new Promise((resolve, reject) => {
      pending.set(id, {
        resolve: (v) => {
          if (signal) signal.removeEventListener("abort", onAbort);
          resolve(v);
        },
        reject: (e) => {
          if (signal) signal.removeEventListener("abort", onAbort);
          genCbs.delete(id);
          reject(e);
        },
      });
      vs().postMessage({ type: "req", id, method: "generate", args: { req } });
    });
  },

  // ---------------------------------------------------------------- 平台能力
  /** 导出 Markdown：主进程 showSaveDialog + fetch + 写文件。 */
  exportMarkdown: (id, title) => call("docs.exportMd", { id, title }),

  /** 主进程 showOpenDialog + 读文件 → {text, title}；取消返回 null。 */
  pickMarkdownFile: () => call("docs.pickFile"),

  /** 主进程模态确认框（webview 无 window.confirm）。 */
  confirm: (message) => call("confirm", { message }),

  // ---------------------------------------------------------------- 持久化
  // webview 生命周期不可靠（dispose 后 state 丢失）→ 经主进程 globalState 持久化。
  loadSettings: () => call("settings.get"),
  saveSettings: (settings) => call("settings.set", { value: settings }),
  loadDraft: () => call("draft.get"),
  persistDraft: (draft) => call("draft.set", { value: draft }),
};

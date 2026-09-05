// 浏览器平台实现：fetch/REST + SSE 直连、localStorage、原生 confirm 与文件对话框。
// 语义与 webview 实现严格一致（同一 platform 接口）。
import {
  apiStatus, apiLoad, apiSkills, apiGenerate, apiStop,
  apiListDocs, apiGetDoc, apiSaveDoc, apiDeleteDoc, apiImportMarkdown,
  exportMarkdownUrl,
} from "../api.js";

const SETTINGS_KEY = "gte.settings";
const DRAFT_KEY = "gte.draft";

/** 惰性创建（或复用 index.html 中预留的）导入文件选择框。 */
function fileInput() {
  let el = document.getElementById("importFile");
  if (!el) {
    el = document.createElement("input");
    el.type = "file";
    el.id = "importFile";
    el.accept = ".md,.markdown,.txt";
    el.hidden = true;
    document.body.appendChild(el);
  }
  return el;
}

export const platform = {
  // ---------------------------------------------------------------- 传输
  apiStatus,
  apiLoad,
  apiSkills,
  apiStop,
  apiListDocs,
  apiGetDoc,
  apiSaveDoc,
  apiDeleteDoc,
  apiImportMarkdown,
  apiGenerate,

  // ---------------------------------------------------------------- 平台能力
  /** 导出 Markdown：浏览器走 <a download> 下载。 */
  async exportMarkdown(id, title) {
    const a = document.createElement("a");
    a.href = exportMarkdownUrl(id);
    a.download = `${title || "document"}.md`;
    a.click();
  },

  /** 选择本地 Markdown 文件 → {text, title}；取消则不 resolve（与旧版一致）。 */
  pickMarkdownFile() {
    return new Promise((resolve) => {
      const input = fileInput();
      const onChange = async () => {
        input.removeEventListener("change", onChange);
        const file = input.files && input.files[0];
        input.value = "";
        if (!file) return resolve(null);
        resolve({
          text: await file.text(),
          title: file.name.replace(/\.(md|markdown|txt)$/i, ""),
        });
      };
      input.addEventListener("change", onChange);
      input.click();
    });
  },

  async confirm(message) {
    return window.confirm(message);
  },

  // ---------------------------------------------------------------- 持久化
  // 异步签名与 webview 实现对齐（app.js 以 Promise 风格调用 .catch 兜底）。
  async loadSettings() {
    try {
      return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
    } catch {
      return null; // 忽略损坏的本地设置
    }
  },

  async saveSettings(settings) {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  },

  async loadDraft() {
    try {
      return JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    } catch {
      return null;
    }
  },

  async persistDraft(draft) {
    if (draft == null) localStorage.removeItem(DRAFT_KEY);
    else localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  },
};

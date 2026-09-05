// 应用编排：文档管理、后端/参数/技能面板、生成控制器（SSE）、快捷键。
// 平台差异（浏览器直连 fetch / VSCode webview postMessage 代理、设置与
// 文件对话框等）统一经 platform 抽象层隔离（platform.js）。
import { platform } from "./platform.js";
import { BlockEditor } from "./editor.js";
import {
  newDocumentBlocks, blocksToMessages, pendingCot, reconcilePpl, avgPpl,
  mergePrefillPpl, skillSystemContent,
} from "./model.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- 状态

const DEFAULT_PARAMS = {
  max_new_tokens: 256,
  do_sample: true,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
  enable_thinking: null,
};

// settings 由 init() 异步装载（webview 端经 postMessage 取 globalState），
// 装载完成前保持默认值；saveSettings 仅在 bindParams 之后由用户交互触发。
const settings = {
  params: { ...DEFAULT_PARAMS },
  context_mode: "chat",
  skills: [],
};
let allSkills = [];            // [{name, description, instructions}]
let currentDoc = { id: "", title: "未命名文档" };
let dirty = false;
let generating = false;
let genCtrl = null;
let ctxPpl = null;
let lastCacheInfo = "";
let previewOpen = false;

const editor = new BlockEditor($("editor"), { onChange: markDirty });

async function loadSettings() {
  const saved = (await platform.loadSettings().catch(() => null)) || {};
  settings.params = { ...DEFAULT_PARAMS, ...(saved.params || {}) };
  settings.context_mode = saved.context_mode || "chat";
  settings.skills = saved.skills || [];
}

function saveSettings() {
  void platform.saveSettings({
    params: settings.params,
    context_mode: settings.context_mode,
    skills: settings.skills,
  }).catch(() => { /* 存储失败不阻断 UI */ });
}

// ---------------------------------------------------------------- 通用 UI

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 3500);
}

function setGenStatus(text) {
  $("genStatus").textContent = text || "";
  updateGenButton();
}

function updateGenButton() {
  $("btnGenerate").textContent = generating
    ? "⏹ 停止 (Ctrl+Enter)"
    : "▶ 生成 (Ctrl+Enter)";
}

// 脏文档草稿（1s 防抖）：webview 关闭无 beforeunload 确认，靠草稿恢复兜底；
// 浏览器端顺带获得崩溃恢复能力。
let draftTimer = null;
function scheduleDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    void platform.persistDraft({
      id: currentDoc.id,
      title: $("docTitle").value,
      blocks: editor.plainBlocks(),
    }).catch(() => { /* 草稿失败不阻断编辑 */ });
  }, 1000);
}

function markDirty() {
  dirty = true;
  $("dirtyDot").hidden = false;
  scheduleDraft();
  if (previewOpen) renderPreview();
}

function markClean() {
  dirty = false;
  $("dirtyDot").hidden = true;
  clearTimeout(draftTimer);
  void platform.persistDraft(null).catch(() => {});
}

// ---------------------------------------------------------------- 文档

async function refreshDocs() {
  try {
    const docs = await platform.apiListDocs();
    const ul = $("docList");
    ul.textContent = "";
    for (const d of docs) {
      const li = document.createElement("li");
      if (d.id === currentDoc.id) li.className = "current";
      const title = document.createElement("span");
      title.className = "doc-title";
      title.textContent = d.title;
      title.title = `${d.title}（更新于 ${d.updated_at}）`;
      const del = document.createElement("button");
      del.className = "doc-del";
      del.textContent = "×";
      del.title = "删除文档";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!(await platform.confirm(`删除文档「${d.title}」？`))) return;
        await platform.apiDeleteDoc(d.id);
        if (d.id === currentDoc.id) newDoc();
        await refreshDocs();
      });
      li.appendChild(title);
      li.appendChild(del);
      li.addEventListener("click", () => void openDoc(d.id));
      ul.appendChild(li);
    }
  } catch (e) {
    $("docList").textContent = "";
    const li = document.createElement("li");
    li.className = "hint";
    li.textContent = `文档列表加载失败：${e.message}`;
    $("docList").appendChild(li);
  }
}

async function confirmDiscard() {
  return !dirty || platform.confirm("当前文档有未保存修改，放弃并继续？");
}

async function openDoc(id) {
  if (!(await confirmDiscard())) return;
  try {
    const doc = await platform.apiGetDoc(id);
    currentDoc = { id: doc.id, title: doc.title };
    $("docTitle").value = doc.title;
    editor.setBlocks((doc.blocks || []).map((b) => ({ ...b })));
    markClean();
    if (previewOpen) renderPreview();
    await refreshDocs();
  } catch (e) {
    toast(`打开文档失败：${e.message}`, true);
  }
}

function newDoc() {
  if (!confirmDiscard()) return;
  currentDoc = { id: "", title: "未命名文档" };
  $("docTitle").value = currentDoc.title;
  editor.setBlocks(newDocumentBlocks());
  markClean();
  if (previewOpen) renderPreview();
  void refreshDocs();
}

async function saveDoc() {
  try {
    const title = $("docTitle").value.trim() || "未命名文档";
    const res = await platform.apiSaveDoc({
      id: currentDoc.id,
      title,
      blocks: editor.plainBlocks(),
    });
    currentDoc = { id: res.id, title };
    markClean();
    await refreshDocs();
    toast("已保存");
  } catch (e) {
    toast(`保存失败：${e.message}`, true);
  }
}

// 导入 / 导出 Markdown（浏览器：本地文件下载；webview：VSCode 文件对话框）
async function doImport() {
  let picked = null;
  try {
    picked = await platform.pickMarkdownFile();
  } catch (e) {
    toast(`导入失败：${e.message}`, true);
    return;
  }
  if (!picked) return;
  if (!(await confirmDiscard())) return;
  try {
    const res = await platform.apiImportMarkdown(picked.text, picked.title);
    await openDoc(res.id);
    toast("已导入 Markdown 并转为块文档");
  } catch (err) {
    toast(`导入失败：${err.message}`, true);
  }
}

async function doExport() {
  if (!currentDoc.id) await saveDoc();
  if (!currentDoc.id) return;
  try {
    await platform.exportMarkdown(currentDoc.id, currentDoc.title || "document");
  } catch (e) {
    toast(`导出失败：${e.message}`, true);
  }
}

$("btnImport").addEventListener("click", () => void doImport());
$("btnExport").addEventListener("click", () => void doExport());

$("btnNew").addEventListener("click", newDoc);
$("btnSave").addEventListener("click", () => void saveDoc());
$("docTitle").addEventListener("input", markDirty);
window.addEventListener("beforeunload", (e) => {
  if (dirty) e.preventDefault();
});

// ---------------------------------------------------------------- 后端面板

function applyModeRows() {
  const mode = $("mode").value;
  $("localRow").hidden = mode !== "local";
  $("llamaRow").hidden = $("llamaGpuRow").hidden = $("llamaCtxRow").hidden = mode !== "llamacpp";
  $("apiBaseUrlRow").hidden = $("apiKeyRow").hidden = $("apiModelRow").hidden = mode !== "api";
}
$("mode").addEventListener("change", applyModeRows);
applyModeRows();

$("btnLoad").addEventListener("click", async () => {
  const mode = $("mode").value;
  const body = { mode };
  if (mode === "api") {
    body.base_url = $("baseUrl").value.trim();
    body.api_key = $("apiKey").value;
    body.model = $("apiModel").value.trim();
    if (!body.base_url || !body.model) {
      toast("Base URL 与模型名不能为空", true);
      return;
    }
  } else if (mode === "llamacpp") {
    body.model_path = $("llamaPath").value.trim();
    body.n_gpu_layers = Number($("nGpuLayers").value);
    body.n_ctx = Number($("nCtx").value);
    if (!body.model_path) {
      toast("GGUF 路径不能为空", true);
      return;
    }
  } else {
    body.model_path = $("modelPath").value.trim();
    if (!body.model_path) {
      toast("模型路径不能为空", true);
      return;
    }
  }
  try {
    await platform.apiLoad(body);
    toast("已提交加载请求，正在后台加载…");
    void watchLoadResult();
  } catch (e) {
    toast(`加载失败：${e.message}`, true);
  }
});

// /api/load 异步后台任务：快速轮询直至 loading 结束，弹出成败通知
async function watchLoadResult() {
  const deadline = Date.now() + 180000;
  let sawLoading = false;
  while (Date.now() < deadline) {
    let st = null;
    try {
      st = await platform.apiStatus();
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }
    applyStatus(st);
    if (st.loading) sawLoading = true;
    if (sawLoading && !st.loading) {
      toast(st.message, !st.loaded);
      return;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

function applyStatus(st) {
  $("statusLine").textContent = st.message || "";
  $("btnGenerate").disabled = !st.loaded && !generating;
}

async function pollStatus() {
  try {
    applyStatus(await platform.apiStatus());
  } catch {
    $("statusLine").textContent = "未连接（服务未启动？）";
    $("btnGenerate").disabled = !generating;
  }
}
setInterval(() => void pollStatus(), 5000);

// ---------------------------------------------------------------- 参数面板

function bindParams() {
  $("ctxMode").value = settings.context_mode;
  $("paramThink").value =
    settings.params.enable_thinking === true ? "on"
    : settings.params.enable_thinking === false ? "off" : "";
  $("paramMax").value = settings.params.max_new_tokens;
  $("paramTemp").value = settings.params.temperature;
  $("paramTopP").value = settings.params.top_p;
  $("paramTopK").value = settings.params.top_k;
  $("paramRp").value = settings.params.repetition_penalty;
  $("paramSample").checked = !!settings.params.do_sample;

  $("ctxMode").addEventListener("change", () => {
    settings.context_mode = $("ctxMode").value;
    saveSettings();
    if (previewOpen) renderPreview();
  });
  $("paramThink").addEventListener("change", () => {
    settings.params.enable_thinking =
      $("paramThink").value === "on" ? true
      : $("paramThink").value === "off" ? false : null;
    saveSettings();
  });
  for (const [id, key, cast] of [
    ["paramMax", "max_new_tokens", Number],
    ["paramTemp", "temperature", Number],
    ["paramTopP", "top_p", Number],
    ["paramTopK", "top_k", Number],
    ["paramRp", "repetition_penalty", Number],
  ]) {
    $(id).addEventListener("change", () => {
      const v = cast($(id).value);
      if (Number.isFinite(v)) settings.params[key] = v;
      saveSettings();
    });
  }
  $("paramSample").addEventListener("change", () => {
    settings.params.do_sample = $("paramSample").checked;
    saveSettings();
  });
}

// ---------------------------------------------------------------- 技能库

async function refreshSkills() {
  try {
    allSkills = await platform.apiSkills();
  } catch (e) {
    toast(`获取技能库失败：${e.message}`, true);
    return;
  }
  const ul = $("skillList");
  ul.textContent = "";
  if (!allSkills.length) {
    const li = document.createElement("li");
    li.className = "hint";
    li.textContent = "技能库为空（服务端 skills/ 目录无技能）";
    ul.appendChild(li);
    return;
  }
  for (const sk of allSkills) {
    const li = document.createElement("li");
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = settings.skills.includes(sk.name);
    cb.addEventListener("change", () => {
      if (cb.checked) settings.skills.push(sk.name);
      else settings.skills = settings.skills.filter((n) => n !== sk.name);
      saveSettings();
      if (previewOpen) renderPreview();
    });
    label.appendChild(cb);
    const name = document.createElement("span");
    name.textContent = sk.name;
    name.title = sk.description || sk.name;
    label.appendChild(name);
    const pin = document.createElement("button");
    pin.className = "pin secondary";
    pin.type = "button";
    pin.textContent = "固化";
    pin.title = "固化为文档内 system 块（文档自包含、可移植复现）";
    pin.addEventListener("click", () => {
      editor.insertSkillSystemBlock(skillSystemContent(sk));
      markDirty();
      toast(`已固化技能「${sk.name}」为系统块`);
    });
    li.appendChild(label);
    li.appendChild(pin);
    ul.appendChild(li);
  }
}
$("btnRefreshSkills").addEventListener("click", () => void refreshSkills());

/** 启用技能 → 系统上下文（与后端 skills_to_context 格式一致，预览用）。 */
function skillCtxPreview() {
  const enabled = allSkills.filter((s) => settings.skills.includes(s.name));
  if (!enabled.length) return "";
  return enabled
    .map((s) => skillSystemContent(s))
    .join("\n\n");
}

// ---------------------------------------------------------------- 困惑度面板

function updatePplPanel() {
  $("ctxPplVal").textContent = ctxPpl == null ? "—" : ctxPpl.toFixed(2);
  const ppls = [];
  for (const b of editor.getBlocksForPpl()) {
    const seg = b.ppl && b.ppl.token_ppls;
    if (seg) for (const p of seg) if (p !== null && p !== undefined) ppls.push(p);
  }
  const avg = avgPpl(ppls);
  $("avgPplVal").textContent = avg == null ? "—" : avg.toFixed(2);
  $("cacheInfo").textContent = lastCacheInfo || "";
}

// ---------------------------------------------------------------- LLM 视角预览

$("btnPreview").addEventListener("click", () => {
  previewOpen = !previewOpen;
  $("previewPane").hidden = !previewOpen;
  $("btnPreview").classList.toggle("primary", previewOpen);
  if (previewOpen) renderPreview();
});

function renderPreview() {
  const container = $("previewMsgs");
  container.textContent = "";
  const blocks = editor.plainBlocks();
  const active = blocks[blocks.length - 1];
  const activeText = active ? active.content : "";
  const msgs = blocksToMessages(blocks.slice(0, -1), skillCtxPreview());

  if (settings.context_mode !== "chat") {
    const note = document.createElement("div");
    note.className = "pv-note";
    note.textContent = `当前为 ${settings.context_mode} 模式：上下文为平文本拼接（【指令】前缀 / 裸续写），非消息序列。以下仅展示 chat 模式映射。`;
    container.appendChild(note);
  }

  for (const m of msgs) {
    container.appendChild(pvMsg(m.role, m.content));
  }
  const pend = pendingCot(blocks);
  if (pend) {
    const note = document.createElement("div");
    note.className = "pv-note";
    let closed = true;
    for (let i = blocks.length - 2; i >= 0; i--) {
      if (blocks[i].type === "cot") { closed = !!blocks[i].closed; break; }
      if (blocks[i].type !== "generate") break;
    }
    note.textContent = closed
      ? "（思考已结束：闭合思维链回灌，模型接着写正文）"
      : "（思维链未闭合：作为续写头回灌，从断点继续思考）";
    container.appendChild(pvMsg("assistant", pend, true));
    container.appendChild(note);
  }
  if (activeText) {
    const el = pvMsg("assistant", activeText, true);
    el.classList.add("pv-prefix");
    container.appendChild(el);
    const note = document.createElement("div");
    note.className = "pv-note";
    note.textContent = "↑ 活动块 = assistant 续写前缀（模型从此处接着写）";
    container.appendChild(note);
  }
}

function pvMsg(role, content, isPrefix = false) {
  const el = document.createElement("div");
  el.className = `pv-msg pv-${role}` + (isPrefix ? " pv-prefix" : "");
  const r = document.createElement("span");
  r.className = "pv-role";
  r.textContent =
    role === "user" ? "user（指令）" :
    role === "assistant" ? "assistant（生成）" : "system（系统）";
  const c = document.createElement("span");
  c.textContent = content;
  el.appendChild(r);
  el.appendChild(c);
  return el;
}

// ---------------------------------------------------------------- 生成控制器

async function toggleGenerate() {
  if (generating) void stopGenerate();
  else void startGenerate();
}

async function startGenerate() {
  if (generating) return;
  const active = editor.activeBlock();

  // 活动块对齐：内容 strip（与 parse_doc / 服务端语义一致），ppl 重建基线
  const activeText = (active.content || "").trim();
  active.content = activeText;
  const [bt, bp] = reconcilePpl(
    active.ppl ? active.ppl.token_texts : [],
    active.ppl ? active.ppl.token_ppls : [],
    activeText
  );
  active.ppl = bt.length ? { token_texts: bt, token_ppls: bp } : null;
  let baseline = active.ppl
    ? { token_texts: [...bt], token_ppls: [...bp] }
    : { token_texts: [], token_ppls: [] };
  editor.refreshBlock(active);

  // cot 基线：紧邻活动块的思维链（"被中断的思考"；content 即思考正文）
  let cot = editor.cotBlock();
  let cotBase = "";
  let cotPplBase = { token_texts: [], token_ppls: [] };
  if (cot) {
    cotBase = (cot.content || "").trim();
    const [ct, cp] = reconcilePpl(
      cot.ppl ? cot.ppl.token_texts : [],
      cot.ppl ? cot.ppl.token_ppls : [],
      cotBase
    );
    cotPplBase = ct.length ? { token_texts: ct, token_ppls: cp } : cotPplBase;
    cot.ppl = ct.length ? { token_texts: ct, token_ppls: cp } : null;
    editor.refreshBlock(cot);
  }

  const req = {
    blocks: editor.plainBlocks(),
    active_text: activeText,
    skills: settings.skills,
    params: settings.params,
    context_mode: settings.context_mode,
  };

  genCtrl = new AbortController();
  const myCtrl = genCtrl;
  generating = true;
  editor.setEditable(false);
  // 思维链恒回灌（正文是否已开始），prefill 耗时随其长度线性增长 → 提前说明
  const resumeHint =
    cotBase ? `⏳ 预填充 ${cotBase.length} 字思维链…（再按 Ctrl+Enter 停止）`
    : "⏳ 生成中…（再按 Ctrl+Enter 停止）";
  setGenStatus(resumeHint);

  try {
    await platform.apiGenerate(
      req,
      {
        onCtxPpl: (ppl) => {
          ctxPpl = ppl;
          updatePplPanel();
        },
        // prefill 打分：服务端对活动块文本（含手动编辑部分）的逐 token ppl。
        // 手动编辑的文字由此获得真实困惑度着色，而非沿用旧色或灰显
        onPrefillPpl: (d) => {
          const merged = mergePrefillPpl(baseline, activeText, d);
          if (!merged) return;
          baseline = merged;
          if (activeText) {
            active.ppl = { token_texts: [...baseline.token_texts], token_ppls: [...baseline.token_ppls] };
            editor.refreshBlock(active);
          }
          updatePplPanel();
        },
        onUpdate: (u) => {
          if (u.cum_text) {
            active.content = activeText + u.cum_text;
            active.ppl = {
              token_texts: [...baseline.token_texts, ...(u.token_texts || [])],
              token_ppls: [...baseline.token_ppls, ...(u.token_ppls || [])],
            };
            editor.refreshBlock(active);
          }
          const rCum = u.reasoning_cum || "";
          if (rCum) {
            if (!cot) cot = editor.ensureCotBlock();
            cot.content = cotBase + rCum;
            const rTexts = [...cotPplBase.token_texts, ...(u.reasoning_token_texts || [])];
            cot.ppl = rTexts.length
              ? {
                  token_texts: rTexts,
                  token_ppls: [...cotPplBase.token_ppls, ...(u.reasoning_token_ppls || [])],
                }
              : null;
            editor.refreshBlock(cot);
          }
          if (u.cache_info) lastCacheInfo = u.cache_info;
          updatePplPanel();
          if (previewOpen) renderPreview();
          if (u.final) {
            // 模型输出过闭标签 → 标记思考已结束（closed 态，不显示标签）
            if (u.reasoning_closed) editor.markCotClosed();
            setGenStatus("✅ 生成完成" + (u.cache_info ? `｜${u.cache_info}` : ""));
          }
        },
        onError: (msg) => {
          if (myCtrl.signal.aborted) return;
          setGenStatus(`❌ 生成失败：${msg}`);
          toast(`生成失败：${msg}`, true);
        },
      },
      myCtrl.signal
    );
  } catch (err) {
    const aborted = myCtrl.signal.aborted || (err && err.name === "AbortError");
    if (!aborted) {
      setGenStatus(`❌ 生成失败：${err.message || err}`);
      toast(`生成失败：${err.message || err}`, true);
    }
  } finally {
    if (genCtrl === myCtrl) genCtrl = null;
    generating = false;
    editor.setEditable(true);
    // 按钮必须随 generating 复位刷新（正常完成/出错路径此前停在"停止"）
    updateGenButton();
    if (myCtrl.signal.aborted) setGenStatus("⏹ 已停止，生成结果已保留");
    markDirty(); // 生成结果需保存
    updatePplPanel();
    if (previewOpen) renderPreview();
  }
}

async function stopGenerate() {
  setGenStatus("⏹ 正在停止…");
  try {
    await platform.apiStop();
  } catch { /* 服务不可达也继续中止本地流 */ }
  genCtrl?.abort();
}

$("btnGenerate").addEventListener("click", () => void toggleGenerate());
$("btnFinalize").addEventListener("click", () => {
  editor.finalizeActive();
  markDirty();
});

// ---------------------------------------------------------------- 快捷键

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    void toggleGenerate();
  } else if (e.key === "F2") {
    e.preventDefault();
    editor.finalizeActive();
    markDirty();
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    void saveDoc();
  }
});

// ---------------------------------------------------------------- 启动

// settings/草稿异步装载（浏览器 localStorage 同步可读、webview 经主进程
// globalState），装载完成后绑定参数、恢复草稿或新建文档，再拉取远端状态。
async function init() {
  await loadSettings();
  bindParams();

  // 脏文档草稿恢复（webview 关闭无确认、浏览器崩溃兜底）；
  // markClean 时已清除，故存在即代表上次未保存的编辑。
  const draft = await platform.loadDraft().catch(() => null);
  if (draft && Array.isArray(draft.blocks) && draft.blocks.length) {
    currentDoc = { id: draft.id || "", title: draft.title || "未命名文档" };
    $("docTitle").value = currentDoc.title;
    editor.setBlocks(draft.blocks.map((b) => ({ ...b })));
    markClean();
    toast("已恢复上次未保存的草稿");
  } else {
    newDoc();
  }

  void refreshSkills();
  void pollStatus();
  void refreshDocs();
}

void init();

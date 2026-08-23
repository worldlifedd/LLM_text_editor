"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/extension.ts
var extension_exports = {};
__export(extension_exports, {
  activate: () => activate,
  deactivate: () => deactivate
});
module.exports = __toCommonJS(extension_exports);
var vscode4 = __toESM(require("vscode"));

// src/server.ts
var fs = __toESM(require("fs"));
var path = __toESM(require("path"));
var vscode = __toESM(require("vscode"));
var import_child_process = require("child_process");
var child = null;
var output = null;
function log(msg) {
  if (!output)
    output = vscode.window.createOutputChannel("GTE Server");
  output.appendLine(msg);
}
function serverUrl() {
  const cfg = vscode.workspace.getConfiguration("gte");
  return (cfg.get("serverUrl") || "http://127.0.0.1:8907").replace(/\/+$/, "");
}
function serverPort() {
  const m = /:(\d+)\/?$/.exec(serverUrl());
  return m ? parseInt(m[1], 10) : 8907;
}
function findServerScript() {
  const cfg = vscode.workspace.getConfiguration("gte");
  const explicit = cfg.get("serverScript");
  if (explicit)
    return explicit;
  const extRoot = path.dirname(path.dirname(__dirname));
  const candidates = [
    path.join(extRoot, "python", "server.py"),
    path.join(extRoot, "server.py"),
    path.join(extRoot, "..", "server.py"),
    path.join(extRoot, "src", "server.py")
  ];
  const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (ws)
    candidates.unshift(path.join(ws, "server.py"));
  for (const c of candidates) {
    try {
      if (fs.existsSync(c))
        return c;
    } catch {
    }
  }
  return null;
}
async function waitUntilReady(timeoutMs = 6e4) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(2e3) });
      if (r.ok)
        return true;
    } catch {
    }
    await new Promise((res) => setTimeout(res, 500));
  }
  return false;
}
async function ensureServer() {
  const url = serverUrl();
  try {
    const r = await fetch(`${url}/api/status`, { signal: AbortSignal.timeout(2e3) });
    if (r.ok)
      return url;
  } catch {
  }
  const cfg = vscode.workspace.getConfiguration("gte");
  if (!cfg.get("autoStartServer")) {
    throw new Error(
      `\u670D\u52A1\u4E0D\u53EF\u8FBE\uFF08${url}\uFF09\uFF0C\u4E14 gte.autoStartServer \u5DF2\u5173\u95ED\u3002\u8BF7\u5148\u8FD0\u884C\uFF1Apython server.py`
    );
  }
  if (child && !child.killed) {
    throw new Error(`\u670D\u52A1\u542F\u52A8\u4E2D\u2026\uFF08${url}\uFF09`);
  }
  const script = findServerScript();
  if (!script) {
    throw new Error(
      `\u672A\u627E\u5230 server.py\uFF1A\u8BF7\u5728\u8BBE\u7F6E gte.serverScript \u4E2D\u6307\u5B9A\uFF0C\u6216\u628A server.py \u653E\u5165\u5DE5\u4F5C\u533A/\u6269\u5C55\u76EE\u5F55`
    );
  }
  const py = cfg.get("pythonCommand") || "python";
  log(`\u542F\u52A8\u670D\u52A1\uFF1A${py} ${script} --port ${serverPort()}`);
  child = (0, import_child_process.spawn)(py, [script, "--port", String(serverPort())], {
    cwd: path.dirname(script),
    env: { ...process.env, PYTHONUNBUFFERED: "1" }
  });
  child.stdout?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.stderr?.on("data", (d) => log(d.toString().replace(/\n$/, "")));
  child.on("exit", (code) => {
    log(`server.py \u9000\u51FA\uFF0Ccode=${code}`);
    child = null;
  });
  if (await waitUntilReady())
    return url;
  throw new Error(`\u670D\u52A1\u542F\u52A8\u8D85\u65F6\uFF08${url}\uFF09\u3002\u67E5\u770B\u300CGTE Server\u300D\u8F93\u51FA\u9762\u677F\u4E86\u89E3\u8BE6\u60C5`);
}
function getStatus() {
  return fetch(`${serverUrl()}/api/status`, { signal: AbortSignal.timeout(3e3) }).then(
    (r) => r.json()
  );
}

// src/api.ts
async function postJson(api, body, timeoutMs = 8e3) {
  const r = await fetch(`${serverUrl()}${api}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs)
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (j && j.error)
        msg = j.error;
    } catch {
    }
    throw new Error(msg);
  }
  return await r.json();
}
function apiLoad(req) {
  return postJson("/api/load", req);
}
function apiStop() {
  return postJson("/api/stop", {});
}
function apiSkills() {
  return postJson("/api/skills", {});
}
async function apiGenerate(req, handlers, signal) {
  const r = await fetch(`${serverUrl()}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (j && j.error)
        msg = j.error;
    } catch {
    }
    handlers.onError(msg);
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let eventName = "";
  const handleData = (dataStr) => {
    let data;
    try {
      data = JSON.parse(dataStr);
    } catch {
      return;
    }
    if (eventName === "ctx_ppl") {
      handlers.onCtxPpl?.(data.ppl);
    } else if (eventName === "update") {
      const u = data;
      handlers.onUpdate(u);
      if (u.final)
        handlers.onDone?.();
    } else if (eventName === "error") {
      handlers.onError(data.error || "\u751F\u6210\u5931\u8D25");
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done)
      break;
    buf += decoder.decode(value, { stream: true });
    let idx;
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

// src/decorations.ts
var vscode2 = __toESM(require("vscode"));
var LOG_PPL_MAX = Math.log(200);
function pplT(ppl) {
  const t = (Math.log(Math.max(ppl, 1.0001)) - 0) / LOG_PPL_MAX;
  return Math.min(Math.max(t, 0), 1);
}
function pplRgb(ppl) {
  const t = pplT(ppl);
  const anchors = [
    [134, 226, 148],
    [255, 226, 130],
    [255, 118, 108]
  ];
  let a, b, u;
  if (t < 0.5) {
    a = anchors[0];
    b = anchors[1];
    u = t * 2;
  } else {
    a = anchors[1];
    b = anchors[2];
    u = (t - 0.5) * 2;
  }
  return [
    Math.round(a[0] + (b[0] - a[0]) * u),
    Math.round(a[1] + (b[1] - a[1]) * u),
    Math.round(a[2] + (b[2] - a[2]) * u)
  ];
}
var HEAT_BUCKETS = 16;
function bucketColor(ppl) {
  const idx = Math.min(HEAT_BUCKETS - 1, Math.floor(pplT(ppl) * HEAT_BUCKETS));
  const midT = (idx + 0.5) / HEAT_BUCKETS;
  const pplAt = Math.exp(midT * LOG_PPL_MAX);
  const [r, g, b] = pplRgb(pplAt);
  return `rgba(${r},${g},${b},0.45)`;
}
var GRAY_BG = "rgba(160,160,160,0.18)";
var Decorator = class {
  heatTypes = /* @__PURE__ */ new Map();
  grayType = vscode2.window.createTextEditorDecorationType({
    backgroundColor: GRAY_BG
  });
  promptBg = vscode2.window.createTextEditorDecorationType({
    backgroundColor: "rgba(110,150,255,0.10)"
  });
  generateBg = vscode2.window.createTextEditorDecorationType({
    backgroundColor: "rgba(60,190,120,0.10)"
  });
  lockedBg = vscode2.window.createTextEditorDecorationType({
    backgroundColor: "rgba(160,160,160,0.12)",
    before: { contentText: "\u{1F512} ", margin: "0 2px 0 0" }
  });
  heatType(color) {
    let t = this.heatTypes.get(color);
    if (!t) {
      t = vscode2.window.createTextEditorDecorationType({
        backgroundColor: color
      });
      this.heatTypes.set(color, t);
    }
    return t;
  }
  /** 将 token 段映射为热力/灰显装饰（从 startOffset 起）。 */
  tokenSegs(doc, tokenTexts, tokenPpls, startOffset, out) {
    let off = startOffset;
    for (let i = 0; i < tokenTexts.length; i++) {
      const t = tokenTexts[i];
      if (!t)
        continue;
      const end = off + t.length;
      const range = new vscode2.Range(doc.positionAt(off), doc.positionAt(end));
      const p = tokenPpls[i];
      const type = p === null || p === void 0 ? this.grayType : this.heatType(bucketColor(p));
      let arr = out.get(type);
      if (!arr) {
        arr = [];
        out.set(type, arr);
      }
      arr.push(range);
      off = end;
    }
  }
  /** 重绘整个文档的装饰。 */
  apply(editor, parsed, state) {
    const doc = editor.document;
    const heat = /* @__PURE__ */ new Map();
    const prompts = [];
    const generates = [];
    const locked = [];
    for (const blk of parsed.blocks) {
      const range = new vscode2.Range(
        doc.positionAt(blk.contentStart),
        doc.positionAt(blk.contentEnd)
      );
      if (blk.type === "prompt") {
        prompts.push(range);
      } else {
        generates.push(range);
        const fin = state.finalized.get(blk.index);
        const segOk = fin && fin.seg.token_texts.length && "".concat(...fin.seg.token_texts) === blk.content;
        if (segOk) {
          this.tokenSegs(doc, fin.seg.token_texts, fin.seg.token_ppls, blk.contentStart, heat);
        } else {
          this.tokenSegs(doc, [blk.content], [null], blk.contentStart, heat);
        }
      }
      if (state.locked.has(blk.index)) {
        locked.push(range);
      }
    }
    const active = parsed.active;
    if (active) {
      generates.push(
        new vscode2.Range(
          doc.positionAt(active.contentStart),
          doc.positionAt(active.contentEnd)
        )
      );
      if (state.activePpl.token_texts.length) {
        this.tokenSegs(
          doc,
          state.activePpl.token_texts,
          state.activePpl.token_ppls,
          active.contentStart,
          heat
        );
      }
    }
    for (const [type, ranges] of heat)
      editor.setDecorations(type, ranges);
    editor.setDecorations(this.grayType, []);
    editor.setDecorations(this.promptBg, prompts);
    editor.setDecorations(this.generateBg, generates);
    editor.setDecorations(this.lockedBg, locked);
  }
  dispose() {
    for (const t of this.heatTypes.values())
      t.dispose();
    this.heatTypes.clear();
    this.grayType.dispose();
    this.promptBg.dispose();
    this.generateBg.dispose();
    this.lockedBg.dispose();
  }
};
function foldingRanges(parsed, doc) {
  return parsed.blocks.map((b) => {
    const startLine = doc.positionAt(b.blockStart).line;
    let endLine = doc.positionAt(b.blockEnd).line;
    const endPos = doc.positionAt(b.blockEnd);
    if (endPos.character === 0 && endLine > startLine)
      endLine -= 1;
    return new vscode2.FoldingRange(startLine, endLine);
  });
}

// src/docmodel.ts
var BLOCK_RE = /<(prompt|generate)>\s*([\s\S]*?)\s*<\/\1>/g;
function parseDoc(text) {
  const blocks = [];
  BLOCK_RE.lastIndex = 0;
  let m;
  while ((m = BLOCK_RE.exec(text)) !== null) {
    const type = m[1];
    const fullMatch = m[0];
    const matchStart = m.index;
    const content = m[2];
    const contentRel = fullMatch.indexOf(content);
    const contentStart = matchStart + contentRel;
    const contentEnd = contentStart + content.length;
    const closeTag = `</${type}>`;
    const closeRel = fullMatch.lastIndexOf(closeTag);
    const blockEnd = matchStart + closeRel + closeTag.length;
    blocks.push({
      type,
      content,
      contentStart,
      contentEnd,
      blockStart: matchStart,
      blockEnd,
      index: blocks.length
    });
  }
  let active = null;
  if (blocks.length && blocks[blocks.length - 1].type === "generate") {
    active = blocks[blocks.length - 1];
  }
  return { blocks, active };
}
function newDocumentText() {
  return [
    "<prompt>",
    "\u5199\u4E00\u6BB5\u5173\u4E8E\u79CB\u5929\u7684\u6563\u6587\uFF0C100\u5B57\u5DE6\u53F3\u3002",
    "</prompt>",
    "",
    "<generate>",
    "",
    "</generate>",
    ""
  ].join("\n");
}

// src/generation.ts
var vscode3 = __toESM(require("vscode"));

// src/state.ts
var emptyPpl = () => ({ token_texts: [], token_ppls: [] });
function newState() {
  return { activePpl: emptyPpl(), finalized: /* @__PURE__ */ new Map(), locked: /* @__PURE__ */ new Set() };
}
var StateStore = class {
  map = /* @__PURE__ */ new Map();
  get(uri) {
    let s = this.map.get(uri);
    if (!s) {
      s = newState();
      this.map.set(uri, s);
    }
    return s;
  }
  clear(uri) {
    this.map.delete(uri);
  }
};
function reconcileActivePpl(tokenTexts, tokenPpls, base) {
  if (!tokenTexts.length)
    return emptyPpl();
  const cov = tokenTexts.join("");
  if (base.startsWith(cov)) {
    return { token_texts: [...tokenTexts], token_ppls: [...tokenPpls] };
  }
  let k = 0;
  const n = Math.min(cov.length, base.length);
  while (k < n && cov[k] === base[k])
    k++;
  const keptT = [];
  const keptP = [];
  let acc = 0;
  for (let i = 0; i < tokenTexts.length; i++) {
    const t = tokenTexts[i];
    if (acc + t.length <= k) {
      keptT.push(t);
      keptP.push(tokenPpls[i]);
      acc += t.length;
    } else {
      break;
    }
  }
  const rest = base.slice(acc);
  if (rest) {
    keptT.push(rest);
    keptP.push(null);
  }
  return { token_texts: keptT, token_ppls: keptP };
}

// src/generation.ts
var GenerationController = class {
  constructor(state, decorator, cb) {
    this.state = state;
    this.decorator = decorator;
    this.cb = cb;
  }
  ctrl = null;
  generating = false;
  selfEdit = false;
  get isGenerating() {
    return this.generating;
  }
  /** 当前活动编辑器及其文档状态；无则返回 null。 */
  session() {
    const editor = vscode3.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "markdown")
      return null;
    const uri = editor.document.uri.toString();
    return { editor, doc: editor.document, state: this.state.get(uri), uri };
  }
  allPpls(parsed, ds) {
    const out = [];
    for (const b of parsed.blocks) {
      if (b.type !== "generate")
        continue;
      const fin = ds.finalized.get(b.index);
      if (fin) {
        for (const p of fin.seg.token_ppls)
          if (p !== null && p !== void 0)
            out.push(p);
      }
    }
    for (const p of ds.activePpl.token_ppls)
      if (p !== null && p !== void 0)
        out.push(p);
    return out;
  }
  avgPpl(ppls) {
    if (!ppls.length)
      return null;
    let sum = 0;
    for (const p of ppls)
      sum += Math.log(Math.max(p, 1e-9));
    return Math.exp(sum / ppls.length);
  }
  emitPpl(parsed, ds, cacheInfo) {
    const series = this.allPpls(parsed, ds);
    this.cb.notifyPanel({
      type: "ppl",
      ctxPpl: this.ctxPpl,
      avgPpl: this.avgPpl(series),
      series,
      cacheInfo
    });
  }
  ctxPpl = null;
  // ---------------------------------------------------------------- 生成
  async start() {
    const s = this.session();
    if (!s) {
      vscode3.window.showWarningMessage("\u8BF7\u5728 Markdown \u6587\u6863\u4E2D\u6267\u884C GTE \u751F\u6210");
      return;
    }
    if (this.generating) {
      vscode3.window.showWarningMessage("\u5DF2\u6709\u751F\u6210\u4EFB\u52A1\u8FDB\u884C\u4E2D\uFF0C\u8BF7\u5148\u505C\u6B62");
      return;
    }
    const { editor, doc, state } = s;
    this.ctxPpl = null;
    let parsed = parseDoc(doc.getText());
    if (!parsed.active) {
      await editor.edit((e) => e.insert(new vscode3.Position(doc.lineCount, 0), "\n\n<generate>\n\n</generate>"));
      parsed = parseDoc(doc.getText());
      if (!parsed.active) {
        vscode3.window.showErrorMessage("\u65E0\u6CD5\u521B\u5EFA\u751F\u6210\u5757");
        return;
      }
    }
    const activeText = parsed.active.content;
    const seg = reconcileActivePpl(
      state.activePpl.token_texts,
      state.activePpl.token_ppls,
      activeText
    );
    state.activePpl = seg;
    const { params, context_mode, skills } = this.cb.getRequestState();
    const req = {
      // 服务端契约：blocks 尾部须为活动 <generate> 块（active 由服务端从最后一块推导）
      blocks: [
        ...parsed.blocks.map((b) => ({ type: b.type, content: b.content })),
        { type: "generate", content: activeText }
      ],
      active_text: activeText,
      skills,
      params,
      context_mode
    };
    const ctrl = new AbortController();
    this.ctrl = ctrl;
    this.generating = true;
    this.cb.statusBar("\u23F3 \u751F\u6210\u4E2D\u2026\uFF08Ctrl+Alt+Enter \u505C\u6B62\uFF09");
    let lastCum = 0;
    const onUpdate = (u) => {
      const delta = u.cum_text.slice(lastCum);
      lastCum = u.cum_text.length;
      if (delta)
        this.insertDelta(delta);
      const ds = this.session();
      if (ds) {
        const merged = {
          token_texts: [...ds.state.activePpl.token_texts, ...u.token_texts],
          token_ppls: [...ds.state.activePpl.token_ppls, ...u.token_ppls]
        };
        const act = parseDoc(ds.doc.getText()).active;
        const cur = act ? act.content : "";
        ds.state.activePpl = reconcileActivePpl(merged.token_texts, merged.token_ppls, cur);
      }
      const parsedNow = parseDoc(this.docText());
      if (parsedNow.active)
        this.emitPpl(parsedNow, this.state.get(this.currentUri()), u.cache_info || "");
      this.cb.refreshDecorations();
      if (u.final)
        this.finish("\u2705 \u751F\u6210\u5B8C\u6210" + (u.cache_info ? `\uFF5C${u.cache_info}` : ""));
    };
    try {
      await apiGenerate(
        req,
        {
          onCtxPpl: (ppl) => {
            this.ctxPpl = ppl;
            const parsedNow = parseDoc(this.docText());
            if (parsedNow.active)
              this.emitPpl(parsedNow, this.state.get(this.currentUri()), "");
          },
          onUpdate,
          onError: (msg) => {
            if (ctrl.signal.aborted)
              return;
            this.finish(`\u274C \u751F\u6210\u5931\u8D25\uFF1A${msg}`, true);
          }
        },
        ctrl.signal
      );
    } catch (err) {
      const aborted = ctrl.signal.aborted || err?.name === "AbortError";
      if (!aborted)
        this.finish(`\u274C \u751F\u6210\u5931\u8D25\uFF1A${err?.message || err}`, true);
    } finally {
      if (this.ctrl === ctrl)
        this.ctrl = null;
      this.generating = false;
      if (!ctrl.signal.aborted)
        this.cb.statusBar("\u23F9 \u5DF2\u505C\u6B62\uFF0C\u751F\u6210\u7ED3\u679C\u5DF2\u4FDD\u7559");
      this.cb.notifyPanel({ type: "gen", generating: false });
      this.cb.refreshDecorations();
    }
  }
  finish(msg, isError = false) {
    this.cb.statusBar(msg, isError ? "GTE \u751F\u6210\u9519\u8BEF" : void 0);
    this.cb.notifyPanel({ type: "gen", generating: false });
  }
  docText() {
    return vscode3.window.activeTextEditor?.document.getText() ?? "";
  }
  currentUri() {
    return vscode3.window.activeTextEditor?.document.uri.toString() ?? "";
  }
  /** 在活动生成块内容末尾插入增量文本。 */
  insertDelta(delta) {
    const editor = vscode3.window.activeTextEditor;
    if (!editor)
      return;
    const parsed = parseDoc(editor.document.getText());
    const act = parsed.active;
    if (!act)
      return;
    const pos = editor.document.positionAt(act.contentEnd);
    this.selfEdit = true;
    void editor.edit((e) => e.insert(pos, delta), { undoStopBefore: false, undoStopAfter: false }).then(() => {
      this.selfEdit = false;
    });
  }
  async stop() {
    this.cb.statusBar("\u23F9 \u6B63\u5728\u505C\u6B62\u2026");
    try {
      await apiStop();
    } catch {
    }
    this.ctrl?.abort();
  }
  // ---------------------------------------------------------------- 定稿 / 追加 / 锁定
  async finalize() {
    const s = this.session();
    if (!s)
      return;
    const { editor, doc, state } = s;
    const parsed = parseDoc(doc.getText());
    if (!parsed.active) {
      vscode3.window.showWarningMessage("\u5F53\u524D\u6587\u6863\u6CA1\u6709\u6D3B\u52A8\u751F\u6210\u5757");
      return;
    }
    const texts = [...state.activePpl.token_texts];
    const ppls = [...state.activePpl.token_ppls];
    while (texts.length && !texts[0].trim()) {
      texts.shift();
      ppls.shift();
    }
    while (texts.length && !texts[texts.length - 1].trim()) {
      texts.pop();
      ppls.pop();
    }
    if (texts.length) {
      texts[0] = texts[0].replace(/^\s+/, "");
      texts[texts.length - 1] = texts[texts.length - 1].replace(/\s+$/, "");
    }
    if (texts.length && "".concat(...texts) === parsed.active.content) {
      state.finalized.set(parsed.active.index, {
        seg: { token_texts: texts, token_ppls: ppls },
        content: parsed.active.content
      });
    }
    for (const b of parsed.blocks)
      state.locked.add(b.index);
    state.activePpl = emptyPpl();
    await editor.edit((e) => e.insert(new vscode3.Position(doc.lineCount, 0), "\n\n<generate>\n\n</generate>"));
    this.cb.statusBar("\u5DF2\u5B9A\u7A3F\u5F53\u524D\u751F\u6210\u5757\u5E76\u5F00\u542F\u65B0\u5757\uFF08\u524D\u5E8F\u5757\u5DF2\u9501\u5B9A\u6298\u53E0\uFF09");
    this.cb.refreshDecorations();
  }
  async addPrompt() {
    const s = this.session();
    if (!s)
      return;
    await s.editor.edit(
      (e) => e.insert(new vscode3.Position(s.doc.lineCount, 0), "\n\n<prompt>\n\n</prompt>")
    );
    this.cb.statusBar("\u5DF2\u8FFD\u52A0\u63D0\u793A\u8BCD\u5757");
    this.cb.refreshDecorations();
  }
  toggleLock() {
    const s = this.session();
    if (!s)
      return;
    const parsed = parseDoc(s.doc.getText());
    const cursor = s.doc.offsetAt(s.editor.selection.active);
    const blk = parsed.blocks.find((b) => cursor >= b.blockStart && cursor <= b.blockEnd);
    if (!blk) {
      vscode3.window.showWarningMessage("\u5149\u6807\u4E0D\u5728\u4EFB\u4F55\u5757\u5185");
      return;
    }
    if (s.state.locked.has(blk.index))
      s.state.locked.delete(blk.index);
    else
      s.state.locked.add(blk.index);
    this.cb.refreshDecorations();
  }
  // ---------------------------------------------------------------- 编辑同步
  /** 文档内容变化：用户编辑时对账 ppl 并重绘。 */
  onDocChanged(e) {
    if (this.selfEdit)
      return;
    if (e.document.languageId !== "markdown")
      return;
    const uri = e.document.uri.toString();
    const ds = this.state.get(uri);
    const parsed = parseDoc(e.document.getText());
    const act = parsed.active;
    if (act) {
      ds.activePpl = reconcileActivePpl(
        ds.activePpl.token_texts,
        ds.activePpl.token_ppls,
        act.content
      );
    }
    this.cb.refreshDecorations();
  }
  /** 编辑器切换/滚动后重绘。 */
  refresh() {
    this.cb.refreshDecorations();
  }
};

// src/panel.ts
var GtePanelProvider = class {
  constructor(resolveReady, handleMessage) {
    this.resolveReady = resolveReady;
    this.handleMessage = handleMessage;
  }
  static viewType = "gte.panel";
  view;
  lastStatus;
  lastPpl;
  resolveWebviewView(webviewView) {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.html();
    webviewView.webview.onDidReceiveMessage((msg) => {
      void this.handleMessage(msg);
    });
    void this.refresh();
  }
  async refresh() {
    if (!this.view)
      return;
    try {
      const ready = await this.resolveReady();
      this.post({ type: "status", status: ready.status });
      this.post({ type: "skills", skills: ready.skills });
      this.post({ type: "params", params: ready.params, context_mode: ready.context_mode });
    } catch (e) {
      this.post({ type: "status", status: { message: `\u274C ${e.message}` } });
    }
  }
  post(msg) {
    if (this.view)
      void this.view.webview.postMessage(msg);
  }
  html() {
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
  <h3>\u{1F50C} \u540E\u7AEF</h3>
  <div class="section">
    <div class="row"><label>\u6A21\u5F0F</label>
      <select id="mode">
        <option value="local">\u672C\u5730\u6A21\u578B\uFF08transformers\uFF09</option>
        <option value="api">OpenAI \u517C\u5BB9 API</option>
      </select>
    </div>
    <div id="localRow" class="row"><label>\u6A21\u578B\u8DEF\u5F84/ID</label>
      <input type="text" id="modelPath" value="Qwen/Qwen2.5-0.5B-Instruct"></div>
    <div id="apiRow" class="row" style="display:none"><label>base_url</label>
      <input type="text" id="apiBase" value="https://api.openai.com/v1"></div>
    <div id="apiRow2" class="row" style="display:none"><label>API Key</label>
      <input type="password" id="apiKey"></div>
    <div id="apiRow3" class="row" style="display:none"><label>\u6A21\u578B\u540D</label>
      <input type="text" id="apiModel" value="gpt-4o-mini"></div>
    <button id="loadBtn">\u52A0\u8F7D / \u8FDE\u63A5</button>
    <div class="status" id="status">\u672A\u52A0\u8F7D</div>
  </div>

  <h3>\u{1F39B} \u751F\u6210\u53C2\u6570</h3>
  <div class="section">
    <div class="row"><label>max_new_tokens</label><input type="number" id="pMax" value="256" min="16" step="16"></div>
    <div class="row"><label>do_sample</label><input type="checkbox" id="pSample" checked></div>
    <div class="row"><label>temperature</label><input type="number" id="pTemp" value="0.8" min="0.1" max="2" step="0.05"></div>
    <div class="row"><label>top_k</label><input type="number" id="pTopK" value="50" min="1" max="200" step="1"></div>
    <div class="row"><label>top_p</label><input type="number" id="pTopP" value="0.95" min="0.05" max="1" step="0.05"></div>
    <div class="row"><label>rep_penalty</label><input type="number" id="pRep" value="1.1" min="1" max="2" step="0.05"></div>
    <div class="row"><label>\u4E0A\u4E0B\u6587\u6A21\u5F0F</label>
      <select id="pMode">
        <option value="chat">chat\uFF08\u804A\u5929\u6A21\u677F\uFF09</option>
        <option value="prefix">prefix\uFF08\u3010\u6307\u4EE4\u3011/\u3010\u6B63\u6587\u3011\uFF09</option>
        <option value="raw">raw\uFF08\u539F\u6587\u88F8\u62FC\u63A5\uFF09</option>
      </select>
    </div>
    <button id="genBtn">\u25B6 \u751F\u6210\uFF08\u5F53\u524D\u6587\u6863\uFF09</button>
    <button class="secondary" id="stopBtn">\u23F9 \u505C\u6B62</button>
  </div>

  <h3>\u{1F9E9} \u6280\u80FD\u5E93</h3>
  <div class="section">
    <div id="skillsList"><span class="tip">\u6280\u80FD\u5E93\u4E3A\u7A7A</span></div>
    <button class="secondary" id="refreshSkills">\u{1F504} \u5237\u65B0\u6280\u80FD\u5E93</button>
  </div>

  <h3>\u{1F4CA} \u56F0\u60D1\u5EA6\u6307\u6807</h3>
  <div class="section">
    <div class="metric">\u4E0A\u4E0B\u6587\u56F0\u60D1\u5EA6\uFF1A<b id="ctxPpl">\u2014</b></div>
    <div class="metric">\u5E73\u5747\u751F\u6210\u56F0\u60D1\u5EA6\uFF1A<b id="avgPpl">\u2014</b></div>
    <div class="metric" id="cacheInfo" style="font-size:12px"></div>
    <div class="legend">
      <span style="background:#86E294">ppl\u22481</span>
      <span style="background:#FFE282">ppl\u224820</span>
      <span style="background:#FF766C">ppl\u2265200</span>
      <span style="background:rgba(160,160,160,0.5)">\u7070=\u624B\u52A8\u7F16\u8F91</span>
    </div>
    <svg id="chart" viewBox="0 0 520 130" preserveAspectRatio="none"></svg>
    <div class="tip">\u7EFF=\u6A21\u578B\u786E\u5B9A \u2192 \u7EA2=\u6A21\u578B\u56F0\u60D1\uFF08\u5168\u90E8\u751F\u6210\u5757\u7D2F\u8BA1\uFF09</div>
  </div>

<script>
const vscode = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);

function setMode(m) {
  const api = m === "api";
  $("localRow").style.display = api ? "none" : "";
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
    svg.innerHTML = '<text x="10" y="20" fill="#888" font-size="12">\u6682\u65E0\u751F\u6210\u6570\u636E</text>';
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
    $("status").textContent = s.message || "\u672A\u52A0\u8F7D";
    if (s.kind) setMode(s.kind);
  } else if (msg.type === "skills") {
    const box = $("skillsList");
    if (!msg.skills || !msg.skills.length) {
      box.innerHTML = '<span class="tip">\u6280\u80FD\u5E93\u4E3A\u7A7A</span>';
    } else {
      box.innerHTML = msg.skills
        .map((s) => '<label><input type="checkbox" value="' + s.name + '"> ' + s.name +
          (s.description ? ' <span style="color:#888">\u2014 ' + s.description + "</span>" : "") + "</label>")
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
    $("ctxPpl").textContent = msg.ctxPpl == null ? "\u2014\uFF08\u4EC5\u672C\u5730\u6A21\u5F0F\uFF09" : String(msg.ctxPpl);
    $("avgPpl").textContent = msg.avgPpl == null ? "\u2014" : msg.avgPpl.toFixed(2);
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
};

// src/extension.ts
var DEFAULT_PARAMS = {
  max_new_tokens: 256,
  do_sample: true,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1
};
function activate(context) {
  const state = new StateStore();
  const decorator = new Decorator();
  const statusBar = vscode4.window.createStatusBarItem(vscode4.StatusBarAlignment.Left, 100);
  statusBar.command = "gte.showPanel";
  statusBar.text = "GTE: \u672A\u8FDE\u63A5";
  statusBar.show();
  function loadSaved() {
    const saved = context.workspaceState.get("gte.params");
    if (saved)
      return saved;
    const cfg = vscode4.workspace.getConfiguration("gte").get("params");
    return {
      params: cfg ? { ...DEFAULT_PARAMS, ...cfg } : DEFAULT_PARAMS,
      context_mode: cfg?.context_mode || "chat",
      skills: []
    };
  }
  function refreshDecorations() {
    const editor = vscode4.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "markdown")
      return;
    const parsed = parseDoc(editor.document.getText());
    decorator.apply(editor, parsed, state.get(editor.document.uri.toString()));
  }
  let panel = null;
  const gen = new GenerationController(state, decorator, {
    statusBar: (text, tooltip) => {
      statusBar.text = text;
      statusBar.tooltip = tooltip;
    },
    refreshDecorations,
    notifyPanel: (msg) => panel?.post(msg),
    getRequestState: () => {
      const s = loadSaved();
      return { params: s.params, context_mode: s.context_mode, skills: s.skills };
    }
  });
  async function pollStatus() {
    try {
      const st = await getStatus();
      const label = st.loading ? `$(sync~spin) GTE: \u52A0\u8F7D\u4E2D\u2026` : st.loaded ? `$(check) GTE: ${st.kind === "local" ? "\u672C\u5730" : "API"}${st.generating ? "\uFF5C\u751F\u6210\u4E2D\u2026" : ""}` : "GTE: \u672A\u8FDE\u63A5";
      statusBar.text = label;
      statusBar.tooltip = st.message;
      panel?.post({ type: "status", status: st });
    } catch {
      statusBar.text = "GTE: \u672A\u8FDE\u63A5";
    }
  }
  const timer = setInterval(() => void pollStatus(), 1e4);
  const provider = new GtePanelProvider(
    async () => {
      await ensureServer();
      const status = await getStatus();
      const skills = await apiSkills();
      const s = loadSaved();
      return { status, skills, params: s.params, context_mode: s.context_mode };
    },
    async (rawMsg) => {
      const m = rawMsg;
      try {
        await ensureServer();
      } catch (e) {
        vscode4.window.showErrorMessage(e.message);
        return;
      }
      const t = m.type;
      if (t === "load") {
        try {
          await apiLoad({
            mode: m.mode,
            model_path: m.model_path,
            base_url: m.base_url,
            api_key: m.api_key,
            model: m.model
          });
          vscode4.window.showInformationMessage("\u5DF2\u63D0\u4EA4\u52A0\u8F7D\u8BF7\u6C42\uFF0C\u8BF7\u7A0D\u5019\u2026\uFF08\u770B\u4E0B\u65B9\u72B6\u6001\uFF09");
        } catch (e) {
          vscode4.window.showErrorMessage(`\u52A0\u8F7D\u5931\u8D25\uFF1A${e.message}`);
        }
        void pollStatus();
      } else if (t === "generate") {
        const params = m.params;
        const context_mode = m.context_mode || "chat";
        const skills = m.skills || [];
        await context.workspaceState.update("gte.params", { params, context_mode, skills });
        void gen.start();
      } else if (t === "stop") {
        void gen.stop();
      } else if (t === "refreshSkills") {
        try {
          const skills = await apiSkills();
          panel?.post({ type: "skills", skills });
        } catch (e) {
          vscode4.window.showErrorMessage(`\u83B7\u53D6\u6280\u80FD\u5E93\u5931\u8D25\uFF1A${e.message}`);
        }
      }
    }
  );
  panel = provider;
  context.subscriptions.push(
    vscode4.commands.registerCommand("gte.generate", () => void gen.start()),
    vscode4.commands.registerCommand("gte.stop", () => void gen.stop()),
    vscode4.commands.registerCommand("gte.finalizeBlock", () => void gen.finalize()),
    vscode4.commands.registerCommand("gte.addPromptBlock", () => void gen.addPrompt()),
    vscode4.commands.registerCommand("gte.toggleLock", () => gen.toggleLock()),
    vscode4.commands.registerCommand("gte.newDocument", async () => {
      const doc = await vscode4.workspace.openTextDocument({
        language: "markdown",
        content: newDocumentText()
      });
      void vscode4.window.showTextDocument(doc);
    }),
    vscode4.commands.registerCommand(
      "gte.showPanel",
      () => void vscode4.commands.executeCommand("gte.panel.focus")
    )
  );
  context.subscriptions.push(
    vscode4.languages.registerFoldingRangeProvider(
      { language: "markdown" },
      {
        provideFoldingRanges(doc) {
          return foldingRanges(parseDoc(doc.getText()), doc);
        }
      }
    )
  );
  context.subscriptions.push(
    vscode4.window.registerWebviewViewProvider(GtePanelProvider.viewType, provider),
    vscode4.workspace.onDidChangeTextDocument((e) => gen.onDocChanged(e)),
    vscode4.window.onDidChangeActiveTextEditor(() => gen.refresh()),
    vscode4.workspace.onDidCloseTextDocument((d) => state.clear(d.uri.toString())),
    { dispose: () => clearInterval(timer) },
    decorator,
    statusBar
  );
}
function deactivate() {
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  activate,
  deactivate
});

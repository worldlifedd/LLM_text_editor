// 模拟流式逐段插入，验证 parseDoc 的 insertEnd 是正确追加点。
//
// 回归背景：cot 注释的内容由 TOKEN_RE 的 `([\s\S]*?)\s*-->` 捕获，尾部空白
// 会被 `\s*` 吃掉，不计入 content，于是 contentEnd 落在最后一个非空白字符
// 之后。若以 contentEnd 为追加点，每插入一段就会把已有尾部空白挤到新内容
// 之后 —— 表现为思维链里所有换行/空格丢失、末尾堆一大坨空白，且文档内容
// 与服务端的 token 序列错位（困惑度着色随之失效，只剩开头几个 token 有色）。
//
// 用法：node tests/test_stream_insert.mjs

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const bundleDir = path.join(root, ".esbuild-test");

// 总是重新打包：缓存会掩盖源码改动（曾因此拿到没有 insertEnd 的旧模块）
execFileSync(
  process.execPath,
  [
    path.join(root, "vscode-extension/node_modules/esbuild/bin/esbuild"),
    path.join(root, "vscode-extension/src/state.ts"),
    path.join(root, "vscode-extension/src/docmodel.ts"),
    "--bundle", "--format=esm", "--log-level=error",
    `--outdir=${bundleDir}`, "--out-extension:.js=.mjs",
  ],
  { stdio: "inherit" }
);

const { parseDoc, cotBody } = await import(pathToFileURL(path.join(bundleDir, "docmodel.mjs")).href);

function insertAt(text, offset, s) {
  return text.slice(0, offset) + s + text.slice(offset);
}

/** 找到 cot 块（blocks 中 type==="cot" 的第一个） */
function cotOf(text) {
  return parseDoc(text).blocks.find((b) => b.type === "cot") || null;
}

/** 找到活动生成块 */
function activeOf(text) {
  return parseDoc(text).active;
}

let fail = 0;
function check(name, got, want) {
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? "  OK  " : "  FAIL"} ${name}`);
  if (!ok) console.log(`         实际 ${JSON.stringify(got)}\n         期望 ${JSON.stringify(want)}`);
}

// ---------------------------------------------------------------- 思维链逐段追加
console.log("思维链：逐段追加（真实流式是逐 token 的，段落里刻意带上尾部空白）");
{
  // 插件端创建 cot 块时自带思考区开标签（所见即所得）
  const OPEN = "<" + "thi" + "nk>";
  let doc = "<!-- prompt\nP\n-->\n\n<!-- cot\n" + OPEN + "\n-->\n\n<!-- generate -->\n";
  const chunks = ["Thinking", " ", "Process:", "\n\n", "1. ", "**Analyze**", " the ", "topic", "\n"];
  for (const ch of chunks) {
    const cot = cotOf(doc);
    if (!cot) throw new Error("cot 块丢失");
    doc = insertAt(doc, cot.insertEnd, ch);
  }
  // 注：尾部空白按设计不进 content（TOKEN_RE 的 \s*--> 会吃掉），
  // 这正是必须改用 insertEnd 追加的原因——空白在文档里仍在，只是不算 content
  check("cot 内容自包含思考区开标签",
    cotOf(doc).content, OPEN + "\nThinking Process:\n\n1. **Analyze** the topic");
  check("cot 纯思考文本正确（cotBody 剔除标签）",
    cotBody(cotOf(doc).content), "Thinking Process:\n\n1. **Analyze** the topic");
  check("cot 之后仍是 generate 标记",
    parseDoc(doc).blocks.at(-1).type, "generate");
  check("文档里尾部换行仍在",
    cotOf(doc).blockEnd - cotOf(doc).insertEnd, 3);
}

// 对照：用 contentEnd 当插入点会把空白挤到末尾（修复前的行为）
console.log("\n对照：修复前用 contentEnd 作为追加点");
{
  const OPEN = "<" + "thi" + "nk>";
  let doc = "<!-- prompt\nP\n-->\n\n<!-- cot\n" + OPEN + "\n-->\n\n<!-- generate -->\n";
  const chunks = ["Thinking", " ", "Process:", "\n\n", "1. ", "**Analyze**", " the ", "topic", "\n"];
  for (const ch of chunks) {
    const cot = cotOf(doc);
    doc = insertAt(doc, cot.contentEnd, ch);
  }
  const content = cotOf(doc).content;
  console.log(`        content = ${JSON.stringify(content)}`);
  console.log(`        ${content.includes("ThinkingProcess") ? "已复现缺陷：空格丢失" : "未复现"}`);
}

// ---------------------------------------------------------------- 正文逐段追加
console.log("\n正文：逐段追加（含换行分段）");
{
  let doc = "<!-- prompt\nP\n-->\n\n<!-- generate -->\n";
  const chunks = ["秋风起，", "梧", "桐叶落。", "\n\n", "第二段。"];
  for (const ch of chunks) {
    const act = activeOf(doc);
    if (!act) throw new Error("活动块丢失");
    doc = insertAt(doc, act.insertEnd, ch);
  }
  check("正文内容完整", activeOf(doc).content, "秋风起，梧桐叶落。\n\n第二段。");
}

// ------------------------------------------------- 追加点稳定性：连续追加不改变已有内容
console.log("\n追加点稳定性：连续 20 次追加后前缀不变");
{
  let doc = "<!-- cot\nA\n-->\n\n<!-- generate -->\n";
  for (let i = 0; i < 20; i++) {
    const cot = cotOf(doc);
    doc = insertAt(doc, cot.insertEnd, `x${i} `);
  }
  const content = cotOf(doc).content;
  check("前缀保持完整", content.slice(0, 6), "A\nx0 x");
  check("20 段全部追加成功", content.includes("x19"), true);
  check("中间无粘连（每个 xN 后都有空格）",
    /x\d( |$)/.test(content) && !/x\d\d/.test(content.replace(/x1\d/g, "xN")), true);
}

// ------------------------------------------------ 活动块定位（误删 generate 标记时）
console.log("\n活动块定位：末尾无 generate 标记时应虚拟空块、续写从文末开始");
{
  // 用户误删了 <!-- generate -->：cot 后面直接跟正文
  const doc = "<!-- prompt\nP\n-->\n\n<!-- cot\n思考过程\n-->\n\n秋日的午后。";
  const p = parseDoc(doc);
  check("active 是末尾的正文块", p.active.content, "秋日的午后。");
  check("cot 仍能被定位到（倒数第二）", p.blocks.at(-2).type, "cot");
  check("续写点在正文末尾", doc.slice(p.active.insertEnd), "");
}
{
  // 思考未完成、正文尚未开始：无 generate 标记
  const doc = "<!-- prompt\nP\n-->\n\n<!-- cot\n想了一半\n-->";
  const p = parseDoc(doc);
  check("虚拟活动块内容为空", p.active.content, "");
  check("虚拟块位置在文末", p.active.insertEnd, doc.length);
  check("cot 在倒数第二（能被复用而非新建）", p.blocks.at(-2).type, "cot");
  // 续写思维链应追加到已有 cot 内部，而不是新建 cot、更不是插到 cot 之前
  const cot = p.blocks.at(-2);
  check("cot 追加点在其 --&gt; 之前", cot.insertEnd, cot.blockEnd - 3);
  const merged = insertAt(doc, cot.insertEnd, "接着想");
  check("追加进的是同一个 cot 块", cotOf(merged).content, "想了一半\n接着想");
}
{
  // 空文档：不应崩，虚拟块在 0
  const p = parseDoc("");
  check("空文档也有活动块", p.active !== null, true);
  check("空文档活动块在 0", p.active.insertEnd, 0);
}
{
  // 正常有 generate 标记时行为不变
  const doc = "<!-- generate -->\n\n已有正文";
  const p = parseDoc(doc);
  check("有标记时 active 仍是标记后的内容", p.active.content, "已有正文");
}

console.log(fail === 0 ? "\nALL STREAM INSERT TESTS PASSED" : `\n${fail} 项失败`);
process.exit(fail === 0 ? 0 : 1);

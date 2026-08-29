// 验证 docmodel.parseDoc 的 contentStart/contentEnd 定位（构建产物中提取逻辑）
// 格式：<!-- prompt/system/cot ... --> 注释 = 提示词/系统/思维链块；
// <!-- generate --> 标记后的可见文本 = 生成块。

const TOKEN_RE = /<!--\s*(?:(prompt|system|cot)\b(?::\s*|\s+)([\s\S]*?)\s*-->|(generate)\s*-->)/g;
const NOTE_OPEN_RE = /^<!--\s*(prompt|system|cot)\b(?::\s*|\s+)/;

function parse(text) {
  const blocks = [];
  TOKEN_RE.lastIndex = 0;
  let pos = 0;
  let pendingGen = null;
  const closeSegment = (end) => {
    const seg = text.slice(pos, end);
    const content = seg.trim();
    const leadWs = (/^\s*/.exec(seg) || [""])[0].length;
    if (pendingGen) {
      pendingGen.content = content;
      pendingGen.contentStart = pendingGen.blockEnd + leadWs;
      pendingGen.contentEnd = pendingGen.contentStart + content.length;
      pendingGen.blockEnd = end;
      pendingGen = null;
    } else if (content) {
      blocks.push({
        type: "generate", content,
        contentStart: pos + leadWs, contentEnd: pos + leadWs + content.length,
        blockStart: pos, blockEnd: end, index: blocks.length,
      });
    }
  };
  let m;
  while ((m = TOKEN_RE.exec(text)) !== null) {
    closeSegment(m.index);
    const fullMatch = m[0];
    const matchStart = m.index;
    if (m[1]) {
      const openLen = (NOTE_OPEN_RE.exec(fullMatch) || [""])[0].length;
      const type = m[1];
      const content = m[2] ?? "";
      blocks.push({
        type, content,
        contentStart: matchStart + openLen, contentEnd: matchStart + openLen + content.length,
        blockStart: matchStart, blockEnd: matchStart + fullMatch.length, index: blocks.length,
      });
    } else {
      blocks.push({
        type: "generate", content: "",
        contentStart: matchStart + fullMatch.length, contentEnd: matchStart + fullMatch.length,
        blockStart: matchStart, blockEnd: matchStart + fullMatch.length, index: blocks.length,
      });
      pendingGen = blocks[blocks.length - 1];
    }
    pos = matchStart + fullMatch.length;
  }
  closeSegment(text.length);
  return blocks;
}

const cases = {
  "空内容块（默认模板）": "<!-- prompt\n写秋天。\n-->\n\n<!-- generate -->\n",
  "非空内容块": "<!-- generate -->\n已有基线文本\n",
  "首插后状态": "<!-- generate -->\n长城\n",
  "系统块（固化技能）": "<!-- system\n# 技能指令: 中文散文写作\n写优美的散文。\n-->\n\n<!-- generate -->\n",
  "思维链块": "<!-- cot\n先构思结构。\n-->\n\n<!-- generate -->\n秋日。\n",
};
let fail = 0;
for (const [name, text] of Object.entries(cases)) {
  for (const b of parse(text)) {
    // 不变量：切片与 content 一致；插入点后只允许空白加块结束符
    // （prompt 为 -->，generate 为下一标记或文末），保证插在块内
    const slice = text.slice(b.contentStart, b.contentEnd);
    const ctx = text.slice(b.contentEnd, b.contentEnd + 20);
    const inBlock = slice === b.content && /^\s*(-->|<!--|$)/.test(ctx);
    if (!inBlock) fail++;
    console.log(
      `[${inBlock ? "PASS" : "FAIL"}] ${name} ${b.type}: content=${JSON.stringify(b.content)}` +
      ` 定位切片=${JSON.stringify(slice)} 插入点后文=${JSON.stringify(ctx)}`
    );
  }
}
// 模拟连续两次增量插入
let doc = cases["空内容块（默认模板）"];
const gen = ["长城是", "中华民族的象征。"];
for (const d of gen) {
  const act = parse(doc).filter((b) => b.type === "generate").pop();
  doc = doc.slice(0, act.contentEnd) + d + doc.slice(act.contentEnd);
}
const act = parse(doc).filter((b) => b.type === "generate").pop();
const inside = act.content === gen.join("");
console.log(`[${inside ? "PASS" : "FAIL"}] 两次增量插入后块内容 = ${JSON.stringify(act.content)}`);
if (!inside) fail++;
// prompt 注释内容定位（装饰范围须精确覆盖注释内文本）
const pdoc = parse(cases["空内容块（默认模板）"]);
const pb = pdoc.find((b) => b.type === "prompt");
const pok = text0 => text0.slice(pb.contentStart, pb.contentEnd) === pb.content;
const p1 = pok(cases["空内容块（默认模板）"]);
console.log(`[${p1 ? "PASS" : "FAIL"}] prompt 注释内容定位 = ${JSON.stringify(pb.content)}`);
if (!p1) fail++;
process.exit(fail ? 1 : 0);

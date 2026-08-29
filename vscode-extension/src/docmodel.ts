// 文档模型：解析纯 Markdown 文档为带偏移的块列表。
// 提示词块 = <!-- prompt ... --> 注释、系统块 = <!-- system ... --> 注释、
// 思维链块 = <!-- cot ... --> 注释（均渲染不可见）；生成块 = 可见正文，
// 前置 <!-- generate --> 注释作为块边界标记（分隔相邻生成块、保留空活动块）。
// 技能可固化为 system 块写入文档，使文档自包含、可移植复现；思维链
//（推理模型输出）存为 cot 块：Markdown 渲染中隐藏、源码中可编辑。
// 与 Python 端 core.parse_doc 语义保持一致（内容 strip、最后一个 generate 为活动单元）。

export type BlockType = "prompt" | "system" | "cot" | "generate";

export interface Block {
  type: BlockType;
  content: string; // strip 后的内容
  /** 内容起始 offset（紧邻开头空白之后） */
  contentStart: number;
  /** 内容结束 offset（结尾空白之前，即插入点） */
  contentEnd: number;
  /** 整个块（含标记/注释）的起始 offset */
  blockStart: number;
  /** 整个块（含标记/注释与其后内容）的结束 offset */
  blockEnd: number;
  /** 文档内的第几个块（0 起） */
  index: number;
}

export interface ParsedDoc {
  blocks: Block[];
  /** 活动生成单元：最后一个 generate 块（若无则 null） */
  active: Block | null;
}

const TOKEN_RE =
  /<!--\s*(?:(prompt|system|cot)\b(?::\s*|\s+)([\s\S]*?)\s*-->|(generate)\s*-->)/g;
// prompt/system/cot 注释前缀（贪婪匹配到内容首字符，与 TOKEN_RE 分隔符一致）
const NOTE_OPEN_RE = /^<!--\s*(prompt|system|cot)\b(?::\s*|\s+)/;

export function parseDoc(text: string): ParsedDoc {
  const blocks: Block[] = [];
  TOKEN_RE.lastIndex = 0;
  let pos = 0; // 上一 token 结束位置
  let pendingGen: Block | null = null; // generate 标记后等待内容的块

  // 处理上一 token 到 end 之间的可见文本段
  function closeSegment(end: number): void {
    const seg = text.slice(pos, end);
    const content = seg.trim();
    // 内容为空时也跳过前导空白，保证空块插入点不落在标记行首之前
    const leadWs = (/^\s*/.exec(seg) || [""])[0].length;
    if (pendingGen) {
      pendingGen.content = content;
      pendingGen.contentStart = pendingGen.blockEnd + leadWs;
      pendingGen.contentEnd = pendingGen.contentStart + content.length;
      pendingGen.blockEnd = end;
      pendingGen = null;
    } else if (content) {
      blocks.push({
        type: "generate",
        content,
        contentStart: pos + leadWs,
        contentEnd: pos + leadWs + content.length,
        blockStart: pos,
        blockEnd: end,
        index: blocks.length,
      });
    }
  }

  let m: RegExpExecArray | null;
  while ((m = TOKEN_RE.exec(text)) !== null) {
    closeSegment(m.index);
    const fullMatch = m[0];
    const matchStart = m.index;
    if (m[1]) {
      // prompt / system 注释块：内容夹在注释内部
      const openLen = (NOTE_OPEN_RE.exec(fullMatch) || [""])[0].length;
      const type = m[1] as BlockType;
      const content = m[2] ?? "";
      blocks.push({
        type,
        content,
        contentStart: matchStart + openLen,
        contentEnd: matchStart + openLen + content.length,
        blockStart: matchStart,
        blockEnd: matchStart + fullMatch.length,
        index: blocks.length,
      });
    } else {
      // generate 边界标记：内容为其后到下一 token（或文末）的可见文本
      blocks.push({
        type: "generate",
        content: "",
        contentStart: matchStart + fullMatch.length,
        contentEnd: matchStart + fullMatch.length,
        blockStart: matchStart,
        blockEnd: matchStart + fullMatch.length,
        index: blocks.length,
      });
      pendingGen = blocks[blocks.length - 1];
    }
    pos = matchStart + fullMatch.length;
  }
  closeSegment(text.length);

  let active: Block | null = null;
  if (blocks.length && blocks[blocks.length - 1].type === "generate") {
    active = blocks[blocks.length - 1];
  }
  return { blocks, active };
}

/** 序列化为纯 Markdown 文本（与 Python serialize_doc 一致）。 */
export function serializeDoc(
  blocks: { type: BlockType; content: string }[],
  activeText: string
): string {
  const parts: string[] = [];
  for (const b of blocks) {
    const c = (b.content || "").trim();
    if (b.type === "generate") {
      parts.push(`<!-- generate -->\n${c}`);
    } else {
      parts.push(`<!-- ${b.type}\n${c}\n-->`);
    }
  }
  const a = (activeText || "").trim();
  if (a) {
    parts.push(`<!-- generate -->\n${a}`);
  }
  return parts.join("\n\n") + "\n";
}

/** 技能 → system 块内容（与后端 skills_to_context 单技能格式一致）。 */
export function skillSystemContent(skill: {
  name: string;
  description?: string;
  instructions: string;
}): string {
  let header = `# 技能指令: ${skill.name}`;
  if (skill.description) {
    header += `\n# 说明: ${skill.description}`;
  }
  return `${header}\n${skill.instructions}`;
}

/** 默认新文档模板。 */
export function newDocumentText(): string {
  return [
    "<!-- prompt",
    "写一段关于秋天的散文，100字左右。",
    "-->",
    "",
    "<!-- generate -->",
    "",
  ].join("\n");
}

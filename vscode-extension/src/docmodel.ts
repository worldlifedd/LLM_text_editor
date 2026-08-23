// 文档模型：解析带 <prompt>/<generate> XML 标记的 Markdown 为带偏移的块列表。
// 与 Python 端 core.parse_doc 语义保持一致（内容 strip、最后一个 generate 为活动单元）。

export type BlockType = "prompt" | "generate";

export interface Block {
  type: BlockType;
  content: string; // strip 后的内容
  /** 内容起始 offset（紧邻开头空白之后） */
  contentStart: number;
  /** 内容结束 offset（结尾空白之前，即插入点） */
  contentEnd: number;
  /** 整个块（含标签）的起始 offset */
  blockStart: number;
  /** 整个块（含标签）的结束 offset */
  blockEnd: number;
  /** 文档内的第几个块（0 起） */
  index: number;
}

export interface ParsedDoc {
  blocks: Block[];
  /** 活动生成单元：最后一个 generate 块（若无则 null） */
  active: Block | null;
}

const BLOCK_RE = /<(prompt|generate)>\s*([\s\S]*?)\s*<\/\1>/g;

export function parseDoc(text: string): ParsedDoc {
  const blocks: Block[] = [];
  BLOCK_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = BLOCK_RE.exec(text)) !== null) {
    const type = m[1] as BlockType;
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
      index: blocks.length,
    });
  }
  let active: Block | null = null;
  if (blocks.length && blocks[blocks.length - 1].type === "generate") {
    active = blocks[blocks.length - 1];
  }
  return { blocks, active };
}

/** 序列化为标准 Markdown+XML 文本（与 Python serialize_doc 一致）。 */
export function serializeDoc(
  blocks: { type: BlockType; content: string }[],
  activeText: string
): string {
  const parts: string[] = [];
  for (const b of blocks) {
    parts.push(`<${b.type}>\n${b.content.trim()}\n</${b.type}>`);
  }
  if (activeText.trim()) {
    parts.push(`<generate>\n${activeText.trim()}\n</generate>`);
  }
  return parts.join("\n\n") + "\n";
}

/** 默认新文档模板。 */
export function newDocumentText(): string {
  return [
    "<prompt>",
    "写一段关于秋天的散文，100字左右。",
    "</prompt>",
    "",
    "<generate>",
    "",
    "</generate>",
    "",
  ].join("\n");
}

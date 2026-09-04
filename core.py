# -*- coding: utf-8 -*-
"""共享纯逻辑：文档序列化、上下文组装、困惑度聚合与着色。

从 app.py 抽取的与 UI 框架无关的逻辑，供 Gradio 前端（app.py）与
VSCode 插件后端（server.py）复用。
"""
import math
import re

# ==================================================================== 文档序列化
# 纯 Markdown 格式：提示词块 = <!-- prompt ... --> 注释（渲染不可见）；
# 系统块 = <!-- system ... --> 注释（技能等系统级指令，渲染不可见）；
# 思维链块 = <!-- cot ... --> 注释（模型推理过程，渲染不可见、源码可编辑）；
# 生成块 = 可见正文，前置 <!-- generate --> 注释作为块边界标记
# （用于分隔相邻生成块、保留空的活动生成单元）。
# 技能可固化为 system 块写入文档，使文档自包含、可在无该技能的环境复现生成。
_TOKEN_RE = re.compile(
    r"<!--\s*(?:(?:(prompt|system|cot)\b(?::\s*|\s+)(.*?)\s*-->)|(generate)\s*-->)",
    re.S,
)


def serialize_doc(blocks, active_text):
    """历史块 + 活动单元（作为最后一个 generate 块）→ 纯 Markdown 文本。"""
    parts = []
    for blk in blocks:
        c = (blk.get("content") or "").strip()
        t = blk["type"]
        if t == "prompt":
            parts.append(f"<!-- prompt\n{c}\n-->")
        elif t == "system":
            parts.append(f"<!-- system\n{c}\n-->")
        elif t == "cot":
            parts.append(f"<!-- cot\n{c}\n-->")
        else:
            parts.append(f"<!-- generate -->\n{c}")
    a = (active_text or "").strip()
    if a:
        parts.append(f"<!-- generate -->\n{a}")
    return "\n\n".join(parts) + "\n"


def parse_doc(text):
    """解析纯 Markdown 文本 → (blocks, active_text)。

    <!-- prompt/system/cot ... --> 注释 → 对应块；<!-- generate --> 标记后的
    可见文本 → 生成块；紧随注释的裸文本也归入生成块。最后一个 generate 进活动单元。
    """
    blocks = []
    pos = 0
    pending_gen = False  # 上一 token 是 generate 标记，等待其内容
    for m in _TOKEN_RE.finditer(text):
        c = text[pos:m.start()].strip()
        if pending_gen:
            blocks[-1]["content"] = c
            pending_gen = False
        elif c:
            blocks.append({"type": "generate", "content": c})
        if m.group(3):  # generate 边界标记
            blocks.append({"type": "generate", "content": ""})
            pending_gen = True
        else:  # prompt / system / cot 注释
            blocks.append({"type": m.group(1), "content": (m.group(2) or "").strip()})
        pos = m.end()
    c = text[pos:].strip()
    if pending_gen:
        blocks[-1]["content"] = c
    elif c:
        blocks.append({"type": "generate", "content": c})
    active = ""
    if blocks and blocks[-1]["type"] == "generate":
        active = blocks[-1]["content"]
        blocks = blocks[:-1]
    return blocks, active


# ==================================================================== 困惑度
_LOG_PPL_MIN, _LOG_PPL_MAX = 0.0, math.log(200.0)


def ppl_rgb(ppl):
    """log(ppl) 在 [0, log200] clamp 后经 绿→黄→红 三锚点插值，返回 (r,g,b)。"""
    t = (math.log(max(ppl, 1.0001)) - _LOG_PPL_MIN) / (_LOG_PPL_MAX - _LOG_PPL_MIN)
    t = min(max(t, 0.0), 1.0)
    anchors = [(134, 226, 148), (255, 226, 130), (255, 118, 108)]  # 绿 → 黄 → 红
    if t < 0.5:
        a, b, u = anchors[0], anchors[1], t * 2
    else:
        a, b, u = anchors[1], anchors[2], (t - 0.5) * 2
    return tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))


def ppl_color(ppl):
    """ppl → rgba 背景色字符串（Gradio 热力图用）。"""
    rgb = ppl_rgb(ppl)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.45)"


def reconcile_active_ppl(token_texts, token_ppls, base):
    """着色对账：活动块被手动编辑后，仅保留与新文本公共字符前缀内完整
    token 的着色；编辑点及之后的旧文本合并为单个无数据段(ppl=None)。
    返回 (token_texts, token_ppls)，二者拼接覆盖整个 base。"""
    if not token_texts:
        return [], []
    cov = "".join(token_texts)
    if base.startswith(cov):
        # 覆盖是 base 前缀但短于 base（如末尾追加文本）：补一个无数据段，
        # 保证返回值覆盖整个 base——否则与后续段拼接会出现字符空洞
        rest = base[len(cov):]
        if not rest:
            return list(token_texts), list(token_ppls)
        return list(token_texts) + [rest], list(token_ppls) + [None]
    k, n = 0, min(len(cov), len(base))
    while k < n and cov[k] == base[k]:
        k += 1
    kept_t, kept_p, acc = [], [], 0
    for t, p in zip(token_texts, token_ppls):
        if acc + len(t) <= k:  # 仅保留完整落在公共前缀内的 token
            kept_t.append(t)
            kept_p.append(p)
            acc += len(t)
        else:
            break
    rest = base[acc:]
    if rest:
        kept_t.append(rest)
        kept_p.append(None)
    return kept_t, kept_p


def blocks_ppls(blocks, active_ppls):
    """全部生成块的累计 ppl 序列（历史冻结块 + 活动块，不含无数据段）。"""
    ppls = []
    for b in blocks or []:
        if b.get("type") != "generate":
            continue
        seg = b.get("ppl") or {}
        ppls.extend(p for p in (seg.get("token_ppls") or []) if p is not None)
    ppls.extend(p for p in active_ppls if p is not None)
    return ppls


def avg_ppl_of(token_ppls):
    """几何平均困惑度；无数据返回 None。"""
    if not token_ppls:
        return None
    return math.exp(sum(math.log(max(p, 1e-9)) for p in token_ppls) / len(token_ppls))


# ==================================================================== 上下文组装
# 思维链标签（拼接构造，避免字面量被工具链按 HTML 清洗，同 backend.py）
_THINK_OPEN = "<" + "thi" + "nk>"
_THINK_CLOSE = "</" + "thi" + "nk>"


def normalize_cot(block):
    """cot 块 → (思考正文, 是否已结束思考)。

    新格式 content=纯思考正文 + closed 属性（前端按钮切换）；
    旧格式标签写在 content 里（标签在→closed，开标签在→未结束）；
    纯文本（历史文档）按已闭合处理（模型稳定出正文）。
    """
    body = (block.get("content") or "").strip()
    closed = block.get("closed")
    if closed is not None:
        return body, bool(closed)
    if _THINK_OPEN in body:
        i = body.rfind(_THINK_OPEN)
        body = body[i + len(_THINK_OPEN):]
        ci = body.rfind(_THINK_CLOSE)
        if ci >= 0:
            return body[:ci].strip("\n"), True
        return body.strip("\n"), False
    return body, True


def blocks_to_messages(blocks, skill_ctx):
    """文档块 → chat 消息列表。prompt→user、generate→assistant，
    相邻同角色块合并（chat 模板不允许连续同角色消息）。"""
    msgs = []
    if skill_ctx:
        msgs.append({"role": "system", "content": skill_ctx})
    for blk in blocks or []:
        c = blk["content"].strip()
        if not c:
            continue
        role = "user" if blk["type"] == "prompt" else "assistant"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + c
        else:
            msgs.append({"role": role, "content": c})
    return msgs


def build_flat_context(blocks, active_text, skill_ctx, mode):
    """prefix / raw 模式的平文本上下文。"""
    parts = [skill_ctx] if skill_ctx else []
    if mode == "prefix":
        for blk in blocks:
            c = blk["content"].strip()
            if not c:
                continue
            tag = "指令" if blk["type"] == "prompt" else "正文"
            parts.append(f"【{tag}】\n{c}")
    else:  # raw
        parts.extend(b["content"].strip() for b in blocks if b["content"].strip())
    if active_text:
        parts.append(active_text)
    return "\n\n".join(parts)


def extract_system(blocks, skill_ctx):
    """抽取文档内 system 块并与本地技能上下文合并 → (system_ctx, 其余块)。

    system 块（固化技能等）在前，本地技能在后；使文档自包含、
    可在无对应技能的环境中复现生成。
    """
    blocks = blocks or []
    sys_parts = [
        b["content"].strip() for b in blocks
        if b.get("type") == "system" and (b.get("content") or "").strip()
    ]
    if skill_ctx and skill_ctx.strip():
        sys_parts.append(skill_ctx.strip())
    system_ctx = "\n\n".join(sys_parts)
    rest = [b for b in blocks if b.get("type") != "system"]
    return system_ctx, rest


def build_prompt(blocks, active_text, skill_ctx, mode, backend, enable_thinking=None):
    """构造 backend 专属 prompt。

    上下文模式：
    - chat:   系统块/技能→system、提示词块→user 指令、生成块→assistant 回复
              （本地=聊天模板包装文本；API=messages 列表）
    - prefix: 系统块/技能加【指令】前缀、提示词块加【指令】前缀、
              生成块加【正文】前缀后拼接
    - raw:    全部块原文裸拼接（纯续写场景）
    """
    system_ctx, blocks = extract_system(blocks or [], skill_ctx)
    active = (active_text or "").strip()
    # 历史轮次的思维链不进上下文：推理模型会自行重新思考，回灌旧思维链
    # 反而干扰（DeepSeek 官方亦建议后续请求不携带 reasoning_content）。
    # 例外：紧邻活动单元的 cot 属于"当前进行中的 assistant 回复"——无论
    # 正文是否已开始都作为思考区回灌（closed → 模型接着写正文；
    # 未闭合 → 从断点继续思考）。否则续写上下文与首次生成不一致：
    # 已生成文本的困惑度着色漂移、KV 前缀缓存整体失效、模型换思路重写。
    pending_cot = ""
    pending_closed = True
    for b in reversed(blocks):
        if b.get("type") == "cot":
            pending_cot, pending_closed = normalize_cot(b)
            break
        if b.get("type") != "generate":
            break  # 中间隔着 prompt/system，已不是本轮的思考
    blocks = [b for b in blocks if b.get("type") != "cot"]
    skill = system_ctx or ""
    # 带标签的续写头（backend _splice_pending_think / flat 回灌共用）：
    # 闭合→模型直接出正文；未闭合→从断点继续思考
    cot_head = ""
    if pending_cot:
        cot_head = _THINK_OPEN + "\n" + pending_cot
        if pending_closed:
            cot_head += "\n" + _THINK_CLOSE
    if mode == "chat":
        msgs = blocks_to_messages(blocks, skill)
        if not any(m["role"] == "user" for m in msgs):
            return ""  # chat 模式必须有指令，否则退化为无意义模板
        return backend.build_chat_prompt(
            msgs, active, enable_thinking, cot_prefix=cot_head
        )
    # flat 模式：cot_head 是思考区、active 是续写点，顺序 = 块内容 + 思考区 + 正文
    flat = build_flat_context(blocks, "", skill, mode)
    if cot_head:
        flat = f"{flat}\n\n{cot_head}" if flat.strip() else cot_head
    if active:
        flat = f"{flat}\n\n{active}" if flat.strip() else active
    return backend.build_flat_prompt(flat)

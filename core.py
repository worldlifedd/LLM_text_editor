# -*- coding: utf-8 -*-
"""共享纯逻辑：文档序列化、上下文组装、困惑度聚合与着色。

从 app.py 抽取的与 UI 框架无关的逻辑，供 Gradio 前端（app.py）与
VSCode 插件后端（server.py）复用。
"""
import math
import re

# ==================================================================== 文档序列化
def serialize_doc(blocks, active_text):
    """历史块 + 活动单元（作为最后一个 generate 块）→ Markdown + XML 文本。"""
    parts = []
    for blk in blocks:
        tag = "prompt" if blk["type"] == "prompt" else "generate"
        parts.append(f"<{tag}>\n{blk['content'].strip()}\n</{tag}>")
    if active_text.strip():
        parts.append(f"<generate>\n{active_text.strip()}\n</generate>")
    return "\n\n".join(parts) + "\n"


def parse_doc(text):
    """解析 Markdown+XML 文本 → (blocks, active_text)。最后一个 generate 进活动单元。"""
    blocks = []
    for m in re.finditer(r"<(prompt|generate)>\s*(.*?)\s*</\1>", text, re.S):
        blocks.append({"type": m.group(1), "content": m.group(2).strip()})
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
        return list(token_texts), list(token_ppls)
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


def build_prompt(blocks, active_text, skill_ctx, mode, backend):
    """构造 backend 专属 prompt。

    上下文模式：
    - chat:   提示词块→user 指令、生成块→assistant 回复（本地=聊天模板
              包装文本；API=messages 列表）
    - prefix: 提示词块加【指令】前缀、生成块加【正文】前缀后拼接
    - raw:    全部块原文裸拼接（纯续写场景）
    """
    blocks = blocks or []
    active = (active_text or "").strip()
    skill = skill_ctx or ""
    if mode == "chat":
        msgs = blocks_to_messages(blocks, skill)
        if not any(m["role"] == "user" for m in msgs):
            return ""  # chat 模式必须有指令，否则退化为无意义模板
        return backend.build_chat_prompt(msgs, active)
    flat = build_flat_context(blocks, active, skill, mode)
    return backend.build_flat_prompt(flat)

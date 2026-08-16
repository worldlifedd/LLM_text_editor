# -*- coding: utf-8 -*-
"""基于 LLM 的生成式文本编辑器 — Gradio 前端。

文档模型：历史块（提示词/生成，动态渲染）+ 活动生成单元（固定组件，
即文档最后一块生成块，流式生成与手动编辑均在此进行）。
"""
import datetime
import html
import math
import os
import re

import gradio as gr
import pandas as pd

from backend import LocalBackend, OpenAICompatBackend
from skills import SKILLS_DIR, import_skill_file, scan_skills, skills_to_context

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

# 单用户本地应用：两种后端常驻（切换不卸载本地模型），按当前模式取用
_BACKENDS = {"local": LocalBackend(), "api": OpenAICompatBackend()}
_ACTIVE = {"kind": "local"}


def _backend():
    return _BACKENDS[_ACTIVE["kind"]]


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


# ==================================================================== 困惑度可视化
_LOG_PPL_MIN, _LOG_PPL_MAX = 0.0, math.log(200.0)

# 活动生成单元的逐 token 困惑度：token_texts 拼接恒等于活动块当前文本，
# ppls 中 None 表示该段无数据（手动编辑区域），渲染为灰色。定稿
# （on_add_generate）时随块存入 block["ppl"]，覆盖全部生成块。
_ACTIVE_PPL = {"token_texts": [], "token_ppls": []}


def _reconcile_active_ppl(base):
    """着色对账：活动块被手动编辑后，仅保留与新文本公共字符前缀内完整
    token 的着色；编辑点及之后的旧文本合并为单个无数据段(ppl=None)。
    返回 (token_texts, token_ppls)，二者拼接覆盖整个 base。"""
    texts, ppls = _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"]
    if not texts:
        return [], []
    cov = "".join(texts)
    if base.startswith(cov):
        return list(texts), list(ppls)
    k, n = 0, min(len(cov), len(base))
    while k < n and cov[k] == base[k]:
        k += 1
    kept_t, kept_p, acc = [], [], 0
    for t, p in zip(texts, ppls):
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


def ppl_color(ppl):
    """log(ppl) 在 [0, log200] clamp 后经 绿→黄→红 三锚点插值，返回 rgba 字符串。"""
    t = (math.log(max(ppl, 1.0001)) - _LOG_PPL_MIN) / (_LOG_PPL_MAX - _LOG_PPL_MIN)
    t = min(max(t, 0.0), 1.0)
    anchors = [(134, 226, 148), (255, 226, 130), (255, 118, 108)]  # 绿 → 黄 → 红
    if t < 0.5:
        a, b, u = anchors[0], anchors[1], t * 2
    else:
        a, b, u = anchors[1], anchors[2], (t - 0.5) * 2
    rgb = tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.45)"


def _mixed_spans(token_texts, token_ppls):
    """着色/灰显 span 列表：ppl=None 的段灰显（保留普通空格：容器
    pre-wrap 已保留空格，替换成 &nbsp; 会使英文单词间失去断行点）。"""
    spans = []
    for t, p in zip(token_texts, token_ppls):
        if p is None:
            spans.append(f'<span style="{_GRAY_BG}">{html.escape(t)}</span>')
        else:
            spans.append(f'<span style="background:{ppl_color(p)}">{html.escape(t)}</span>')
    return spans


_GRAY_BG = "background:rgba(160,160,160,0.18);color:#777"
_BLOCK_SEP = '<span style="color:#c8c8c8;margin:0 3px;user-select:none">▍</span>'


def _blocks_ppls(blocks):
    """全部生成块的累计 ppl 序列（历史冻结块 + 活动块，不含无数据段）。"""
    ppls = []
    for b in blocks or []:
        if b.get("type") != "generate":
            continue
        seg = b.get("ppl") or {}
        ppls.extend(p for p in (seg.get("token_ppls") or []) if p is not None)
    ppls.extend(p for p in _ACTIVE_PPL["token_ppls"] if p is not None)
    return ppls


def doc_heatmap_html(blocks, active_text):
    """全文档生成文本热力图：历史生成块 + 活动块按 token 困惑度着色。

    有 ppl 数据的部分按绿→红着色；无数据（手动编辑/外部加载）的文本
    以灰色底显示，保持文档全貌可见。
    """
    spans, has_colored = [], False
    for b in blocks or []:
        if b.get("type") != "generate":
            continue
        c = (b.get("content") or "").strip()
        if not c:
            continue
        seg = b.get("ppl") or {}
        texts, ppls = seg.get("token_texts") or [], seg.get("token_ppls") or []
        if spans:
            spans.append(_BLOCK_SEP)
        if texts and ppls and "".join(texts) == c:
            spans.extend(_mixed_spans(texts, ppls))
            if any(p is not None for p in ppls):
                has_colored = True
        else:
            spans.append(f'<span style="{_GRAY_BG}">{html.escape(c)}</span>')
    # 活动生成单元：着色段 + 无数据灰段 + 尚未对齐的尾部灰段
    at = active_text or ""
    if at:
        if spans:
            spans.append(_BLOCK_SEP)
        texts, ppls = _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"]
        covered = "".join(texts)
        if texts and at.startswith(covered):
            spans.extend(_mixed_spans(texts, ppls))
            tail = at[len(covered):]
            if tail:
                spans.append(f'<span style="{_GRAY_BG}">{html.escape(tail)}</span>')
            if any(p is not None for p in ppls):
                has_colored = True
        else:
            spans.append(f'<span style="{_GRAY_BG}">{html.escape(at)}</span>')
    if not spans:
        return "<i style='color:#888'>暂无生成数据。点击「▶ 生成」后此处按 token 困惑度着色：绿=模型确定，红=模型困惑。</i>"
    legend = (
        '<div style="margin-bottom:6px;font-size:12px;color:#666">'
        "困惑度着色（全部生成块）："
        f'<span style="background:{ppl_color(1)}">&nbsp;ppl≈1&nbsp;</span> → '
        f'<span style="background:{ppl_color(20)}">&nbsp;ppl≈20&nbsp;</span> → '
        f'<span style="background:{ppl_color(200)}">&nbsp;ppl≥200&nbsp;</span>'
        f'｜<span style="{_GRAY_BG}">&nbsp;灰=手动编辑/无数据&nbsp;</span>'
        "｜▍=块边界"
        "</div>"
    )
    body = "".join(spans)
    tip = "" if has_colored else (
        '<div style="font-size:12px;color:#888;margin-bottom:4px">'
        "当前无着色数据（文本为手动输入或从文件加载）。</div>"
    )
    return (
        f"{legend}{tip}<div style='white-space:pre-wrap;overflow-wrap:anywhere;"
        f"line-height:1.9;font-size:14px;padding:8px;"
        f"border:1px solid #e0e0e0;border-radius:6px'>{body}</div>"
    )


def empty_plot_df():
    return pd.DataFrame({"index": pd.Series(dtype="int"), "ppl": pd.Series(dtype="float"), "type": pd.Series(dtype="object")})


def plot_data(token_ppls, window=10):
    """gr.LinePlot 数据：逐 token 困惑度 + 滑动平均（DataFrame）。"""
    rows = []
    for i, p in enumerate(token_ppls):
        rows.append({"index": i + 1, "ppl": p, "type": "逐token"})
        lo = max(0, i + 1 - window)
        seg = token_ppls[lo : i + 1]
        avg = math.exp(sum(math.log(max(x, 1e-9)) for x in seg) / len(seg))
        rows.append({"index": i + 1, "ppl": avg, "type": "滑动平均"})
    if not rows:
        return empty_plot_df()
    return pd.DataFrame(rows)


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


# ==================================================================== 事件处理器
def on_load_model(mode, model_path, base_url, api_key, api_model):
    """按模式加载：local=本地模型；api=OpenAI 兼容 API 连接验证。"""
    backend = _BACKENDS["api"] if mode == "api" else _BACKENDS["local"]
    if mode == "api":
        yield {model_status: f"⏳ 正在连接 API：{base_url} …"}
        try:
            backend.load(base_url, api_key, api_model)
            _ACTIVE["kind"] = "api"
            yield {
                model_status: (
                    f"✅ API 已连接：{backend.model} @ {backend.base_url}"
                    "｜logprobs（逐 token 困惑度）将在首次生成时自动探测"
                )
            }
        except Exception as e:  # noqa: BLE001
            yield {model_status: f"❌ API 连接失败：{e}"}
        return
    path = (model_path or "").strip()
    if not path:
        yield {model_status: "❌ 请先填写模型路径或 HuggingFace 模型 ID"}
        return
    yield {model_status: f"⏳ 正在加载模型：{path} （首次会自动下载，请耐心等待）"}
    try:
        backend.load(path)
        _ACTIVE["kind"] = "local"
        yield {
            model_status: (
                f"✅ 已加载：{path}｜设备：{backend.device}｜"
                f"参数量：{sum(p.numel() for p in backend.model.parameters()) / 1e6:.0f}M"
            )
        }
    except Exception as e:  # noqa: BLE001
        yield {model_status: f"❌ 加载失败：{e}"}


def on_generate(blocks, active_text, enabled_skill_names, skills_list,
                max_new_tokens, do_sample, temperature, top_k, top_p,
                repetition_penalty, context_mode):
    backend = _backend()
    if not backend.loaded:
        yield {status_tb: "❌ 模型尚未加载/连接，请先在顶栏完成加载"}
        return

    enabled = [s for s in (skills_list or []) if s.name in (enabled_skill_names or [])]
    skill_ctx = skills_to_context(enabled)
    try:
        prompt = build_prompt(blocks or [], active_text or "", skill_ctx,
                              mode=context_mode or "chat", backend=backend)
    except Exception as e:  # 模型无聊天模板等
        yield {status_tb: f"❌ 上下文构造失败（可切换为 prefix/raw 模式）：{e}"}
        return
    if not prompt or (isinstance(prompt, str) and not prompt.strip()):
        yield {status_tb: "❌ 文档为空：请先添加提示词块并输入内容"}
        return

    base = active_text or ""
    # 对账：编辑点之前的着色保留，编辑区域及之后合并为无数据段(None)
    seg_t, seg_p = _reconcile_active_ppl(base)
    _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"] = seg_t, seg_p
    blocks = blocks or []

    n_skills = len(enabled)
    all_ppls = _blocks_ppls(blocks)
    yield {
        status_tb: f"⏳ 正在计算上下文困惑度…（启用技能 {n_skills} 个）",
        avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
        ppl_plot: plot_data(all_ppls),
        heatmap_md: doc_heatmap_html(blocks, base),
    }

    try:
        # 上下文困惑度仅本地模式可用（API 不返回 prompt token 概率）
        ctx_ppl_val = None
        if backend.kind == "local":
            ctx_ppl = backend.compute_context_ppl(prompt)
            ctx_ppl_val = round(ctx_ppl, 2) if ctx_ppl == ctx_ppl else None
        yield {
            ctx_ppl_num: ctx_ppl_val,
            status_tb: "⏳ 生成中…（可随时点击「⏹ 停止」后手动编辑）",
        }

        params = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        )
        for upd in backend.generate_stream(prompt, **params):
            full_text = base + upd.cum_text
            # 累积：对账后的基线覆盖段 + 本轮新 token（拼接恒等于 full_text）
            _ACTIVE_PPL["token_texts"] = seg_t + list(upd.token_texts)
            _ACTIVE_PPL["token_ppls"] = seg_p + list(upd.token_ppls)
            all_ppls = _blocks_ppls(blocks)
            cache_note = getattr(backend, "last_cache_info", "")
            yield {
                active_cell_tb: full_text,
                avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
                ppl_plot: plot_data(all_ppls),
                heatmap_md: doc_heatmap_html(blocks, full_text),
                ctx_ppl_num: ctx_ppl_val,
                status_tb: (
                    f"{'✅ 生成完成' if upd.final else '⏳ 生成中'}…"
                    f"｜本轮 token：{len(upd.token_ppls)}"
                    f"｜累计着色：{len(all_ppls)}｜"
                    f"平均困惑度：{avg_ppl_of(all_ppls) or 0:.2f}"
                    + (f"｜{cache_note}" if cache_note and backend.kind == "local" else "")
                ),
            }
    except Exception as e:  # noqa: BLE001
        yield {status_tb: f"❌ 生成失败：{e}"}


def on_stop():
    _backend().stop()
    return {status_tb: "⏹ 已请求停止，等待模型收尾…（生成结果将保留，可手动编辑后续写）"}


def on_add_prompt(blocks):
    blocks = list(blocks or [])
    blocks.append({"type": "prompt", "content": ""})
    return {
        blocks_state: blocks,
        status_tb: "已添加提示词块（编辑后失焦即保存）",
    }


def on_add_generate(blocks, active_text):
    blocks = list(blocks or [])
    c = (active_text or "").strip()
    blk = {"type": "generate", "content": c}
    if _ACTIVE_PPL["token_texts"]:
        # 着色数据随块冻结保留；块内容为 strip 后文本，需先对齐首尾空白
        seg_t = list(_ACTIVE_PPL["token_texts"])
        seg_p = list(_ACTIVE_PPL["token_ppls"])
        while seg_t and not seg_t[0].strip():
            seg_t.pop(0), seg_p.pop(0)
        while seg_t and not seg_t[-1].strip():
            seg_t.pop(), seg_p.pop()
        if seg_t:
            seg_t[0] = seg_t[0].lstrip()
            seg_t[-1] = seg_t[-1].rstrip()
        if seg_t and "".join(seg_t) == c:
            blk["ppl"] = {"token_texts": seg_t, "token_ppls": seg_p}
    blocks.append(blk)
    _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"] = [], []
    all_ppls = _blocks_ppls(blocks)
    return {
        blocks_state: blocks,
        active_cell_tb: "",
        avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
        ppl_plot: plot_data(all_ppls),
        heatmap_md: doc_heatmap_html(blocks, ""),
        status_tb: "已定稿当前生成块并开启新块（困惑度数据随块保留）",
    }


def on_save(blocks, active_text):
    os.makedirs(SAVE_DIR, exist_ok=True)
    content = serialize_doc(blocks or [], active_text or "")
    fname = f"doc_{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    path = os.path.join(SAVE_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {
        download_file: path,
        status_tb: f"💾 已保存：{fname}（可点击右侧文件下载）",
    }


def on_load_doc(file):
    if file is None:
        return {status_tb: "❌ 请先选择 .md 文件"}
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
        blocks, active = parse_doc(text)
        _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"] = [], []
        return {
            blocks_state: blocks,
            active_cell_tb: active,
            avg_ppl_num: None,
            ppl_plot: empty_plot_df(),
            heatmap_md: doc_heatmap_html(blocks, active),
            status_tb: f"📂 已加载 {len(blocks)} 个历史块，活动生成单元 {len(active)} 字（外部加载文本无着色数据）",
        }
    except Exception as e:  # noqa: BLE001
        return {status_tb: f"❌ 加载失败：{e}"}


# ------------------------------------------------------------------ 技能库
def _skills_info_html(skills):
    if not skills:
        return "<i style='color:#888'>技能库为空。将 SKILL.md（Anthropic 风格 frontmatter）"
        "或普通 .md 提示词文件放入 ./skills/ 目录，或直接上传。</i>"
    items = "".join(
        f"<li><b>{html.escape(s.name)}</b>"
        + (f" — {html.escape(s.description)}" if s.description else "")
        + "</li>"
        for s in skills
    )
    return f"<ul style='margin:4px 0 0 16px;font-size:12px'>{items}</ul>"


def on_refresh_skills():
    skills = scan_skills()
    return {
        skills_state: skills,
        skills_check: gr.CheckboxGroup(choices=[s.name for s in skills]),
        skills_info: _skills_info_html(skills),
        status_tb: f"🔄 技能库已刷新：共 {len(skills)} 个技能",
    }


def on_upload_skill(file):
    if file is None:
        return {status_tb: "❌ 请先选择 .md 技能文件"}
    try:
        dst = import_skill_file(file.name)
        skills = scan_skills()
        return {
            skills_state: skills,
            skills_check: gr.CheckboxGroup(choices=[s.name for s in skills]),
            skills_info: _skills_info_html(skills),
            status_tb: f"✅ 技能已导入：{os.path.basename(dst)}",
        }
    except Exception as e:  # noqa: BLE001
        return {status_tb: f"❌ 导入失败：{e}"}


# ==================================================================== UI
_initial_skills = scan_skills()

with gr.Blocks(title="生成式文本编辑器") as demo:
    gr.Markdown("# 🖋 生成式文本编辑器\n类 Jupyter 分块：提示词块引导生成，生成块流式续写、可暂停可手动编辑。")

    # 顶栏：后端模式 + 模型加载
    backend_mode_rd = gr.Radio(
        choices=[("🏠 本地模型（transformers）", "local"),
                 ("☁️ OpenAI 兼容 API", "api")],
        value="local",
        label="后端模式",
        scale=1,
    )
    with gr.Group() as local_cfg_group:
        with gr.Row():
            model_path_tb = gr.Textbox(
                label="模型路径 / HuggingFace ID",
                value="Qwen/Qwen2.5-0.5B-Instruct",
                scale=3,
            )
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
    with gr.Group(visible=False) as api_cfg_group:
        gr.Markdown(
            "<span style='font-size:12px;color:#888'>兼容 OpenAI / DeepSeek / "
            "Qwen(DashScope compatible-mode) / GLM / Kimi，以及 vLLM / Ollama / "
            "llama.cpp 本地服务。base_url 填写到 /v1 层级，"
            "如 https://api.deepseek.com/v1 或 http://localhost:11434/v1</span>"
        )
        with gr.Row():
            api_base_tb = gr.Textbox(
                label="base_url", value="https://api.openai.com/v1", scale=2,
            )
            api_key_tb = gr.Textbox(
                label="API Key", type="password", value="", scale=2,
            )
            api_model_tb = gr.Textbox(
                label="模型名", value="gpt-4o-mini", scale=1,
            )
            api_load_btn = gr.Button("连接 API", variant="primary", scale=1)
    model_status = gr.Textbox(label="模型状态", value="未加载", interactive=False)

    with gr.Row():
        # ---------------------------------------------------------- 左侧面板
        with gr.Column(scale=1):
            with gr.Accordion("🎛 生成参数", open=True):
                context_mode_rd = gr.Radio(
                    choices=[
                        ("chat（聊天模板·Instruct 模型推荐）", "chat"),
                        ("prefix（【指令】/【正文】标记·base 模型）", "prefix"),
                        ("raw（原文裸拼接·纯续写）", "raw"),
                    ],
                    value="chat",
                    label="上下文模式（提示词块如何呈现给模型）",
                )
                max_new_tokens_sl = gr.Slider(16, 2048, value=256, step=16,
                                              label="max_new_tokens（最大生成 token 数）")
                do_sample_cb = gr.Checkbox(value=True, label="do_sample（采样；关闭则贪心解码）")
                temperature_sl = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature")
                top_k_sl = gr.Slider(1, 200, value=50, step=1, label="top_k")
                top_p_sl = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="top_p")
                rep_pen_sl = gr.Slider(1.0, 2.0, value=1.1, step=0.05,
                                       label="repetition_penalty")
                gr.Markdown(
                    "<span style='font-size:12px;color:#888'>API 模式：仅 "
                    "max_new_tokens/temperature/top_p 生效（映射为 max_tokens 等"
                    "标准参数），top_k/repetition_penalty 为本地专属；"
                    "do_sample 关闭时 temperature 置 0。</span>"
                )

            with gr.Accordion("📊 困惑度指标", open=True):
                ctx_ppl_num = gr.Number(
                    label="上下文困惑度（提示词准确度，越低越好·仅本地模式）",
                    value=None, precision=2,
                )
                avg_ppl_num = gr.Number(label="平均生成困惑度（实时·全部生成块累计）", value=None, precision=2)
                gr.Markdown(
                    "<span style='font-size:12px;color:#888'>困惑度反映模型对文本的"
                    "“意外程度”：上下文困惑度高说明提示词对模型而言生僻/混乱；"
                    "生成困惑度低说明输出在模型预期之内。API 模式下逐 token 困惑度"
                    "依赖服务端 logprobs（OpenAI/vLLM 支持，自动探测），不支持时"
                    "自动隐藏。</span>"
                )

            with gr.Accordion("🧩 技能库 Skill Library", open=False):
                skills_state = gr.State(_initial_skills)
                skills_check = gr.CheckboxGroup(
                    choices=[s.name for s in _initial_skills],
                    label="启用的技能（作为系统指令拼入生成上下文）",
                )
                skills_info = gr.HTML(_skills_info_html(_initial_skills))
                with gr.Row():
                    refresh_skills_btn = gr.Button("🔄 刷新技能库", size="sm")
                upload_skill_file = gr.File(
                    label="上传 skill（.md）", file_types=[".md"], height=80
                )
                gr.Markdown(
                    f"<span style='font-size:11px;color:#888'>技能目录：{SKILLS_DIR}"
                    "（支持 Anthropic 风格 SKILL.md 与普通 .md 提示词）</span>"
                )

        # ---------------------------------------------------------- 文档区
        with gr.Column(scale=3):
            gr.Markdown("#### 📄 文档")
            blocks_state = gr.State(
                [{"type": "prompt", "content": "写一段关于秋天的散文开头，100字左右。"}]
            )

            @gr.render(inputs=[blocks_state])
            def render_blocks(blocks):
                for i, blk in enumerate(blocks or []):
                    is_prompt = blk["type"] == "prompt"
                    label = ("📝 提示词块" if is_prompt else "⚙️ 生成块") + f" #{i + 1}"
                    with gr.Group():
                        with gr.Row():
                            tb = gr.Textbox(
                                value=blk["content"],
                                label=label,
                                lines=4,
                                interactive=True,
                                scale=20,
                            )
                            del_btn = gr.Button("🗑 删除", size="sm", scale=1)

                        def _update(blocks_list, text, idx=i):
                            blocks_list = [dict(b) for b in (blocks_list or [])]
                            if idx >= len(blocks_list) or blocks_list[idx]["content"] == text:
                                return gr.skip()  # 无变化：跳过，避免触发重渲染
                            blocks_list[idx]["content"] = text
                            blocks_list[idx].pop("ppl", None)  # 内容已改，着色数据失效
                            return blocks_list

                        def _delete(blocks_list, idx=i):
                            return [b for j, b in enumerate(blocks_list or []) if j != idx]

                        # 用 blur 而非 change：change 在 IME 按 Enter 确认候选词时
                        # 也会触发，导致 blocks_state 更新 → @gr.render 销毁重建
                        # 正在输入的组件 → SSE 响应解析失败（Could not parse
                        # server response）。blur 仅失焦时触发，规避该竞态。
                        tb.blur(_update, inputs=[blocks_state, tb], outputs=[blocks_state])
                        del_btn.click(_delete, inputs=[blocks_state], outputs=[blocks_state])

            active_cell_tb = gr.Textbox(
                label="⚙️ 生成块（当前·最后一块）— LLM 流式输出于此，暂停后可手动编辑，再次生成将续写",
                lines=8,
                placeholder="点击「▶ 生成」后，模型在此流式续写…",
                interactive=True,
            )

            with gr.Accordion("📈 困惑度分析（随生成实时更新）", open=False):
                ppl_plot = gr.LinePlot(
                    value=empty_plot_df(),
                    x="index",
                    y="ppl",
                    color="type",
                    color_title="曲线",
                    y_title="困惑度",
                    x_title="token 序号（全部生成块累计）",
                    height=240,
                )
                heatmap_md = gr.HTML(doc_heatmap_html([], ""))

            # 工具栏
            with gr.Row():
                add_prompt_btn = gr.Button("＋ 提示词块", size="sm")
                add_generate_btn = gr.Button("＋ 生成块（定稿当前，开新块）", size="sm")
                generate_btn = gr.Button("▶ 生成", variant="primary")
                stop_btn = gr.Button("⏹ 停止", variant="stop")
            with gr.Row():
                save_btn = gr.Button("💾 保存 .md", size="sm")
                upload_doc_file = gr.File(label="📂 加载 .md", file_types=[".md"], height=80)
                download_file = gr.File(label="下载区（保存后出现）", interactive=False, height=80)

            status_tb = gr.Textbox(label="状态", value="就绪。请先加载模型。", interactive=False)

    # ---------------------------------------------------------------- 事件接线
    _load_inputs = [backend_mode_rd, model_path_tb, api_base_tb, api_key_tb, api_model_tb]
    load_btn.click(on_load_model, inputs=_load_inputs, outputs=[model_status])
    api_load_btn.click(on_load_model, inputs=_load_inputs, outputs=[model_status])

    def _switch_backend_mode(mode):
        return (gr.Group(visible=mode == "local"), gr.Group(visible=mode == "api"))

    backend_mode_rd.change(_switch_backend_mode, inputs=[backend_mode_rd],
                           outputs=[local_cfg_group, api_cfg_group])

    generate_btn.click(
        on_generate,
        inputs=[
            blocks_state, active_cell_tb, skills_check, skills_state,
            max_new_tokens_sl, do_sample_cb, temperature_sl, top_k_sl,
            top_p_sl, rep_pen_sl, context_mode_rd,
        ],
        outputs=[
            active_cell_tb, avg_ppl_num, ppl_plot, heatmap_md,
            ctx_ppl_num, status_tb,
        ],
    )
    stop_btn.click(on_stop, inputs=None, outputs=[status_tb])
    add_prompt_btn.click(on_add_prompt, inputs=[blocks_state],
                         outputs=[blocks_state, status_tb])
    add_generate_btn.click(on_add_generate, inputs=[blocks_state, active_cell_tb],
                           outputs=[blocks_state, active_cell_tb,
                                    avg_ppl_num, ppl_plot, heatmap_md, status_tb])
    save_btn.click(on_save, inputs=[blocks_state, active_cell_tb],
                   outputs=[download_file, status_tb])
    upload_doc_file.change(on_load_doc, inputs=[upload_doc_file],
                           outputs=[blocks_state, active_cell_tb,
                                    avg_ppl_num, ppl_plot, heatmap_md, status_tb])
    refresh_skills_btn.click(on_refresh_skills, inputs=None,
                             outputs=[skills_state, skills_check, skills_info, status_tb])
    upload_skill_file.change(on_upload_skill, inputs=[upload_skill_file],
                             outputs=[skills_state, skills_check, skills_info, status_tb])

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), inbrowser=True)

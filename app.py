# -*- coding: utf-8 -*-
"""基于 LLM 的生成式文本编辑器 — Gradio 前端。

文档模型：历史块（提示词/生成，动态渲染）+ 活动生成单元（固定组件，
即文档最后一块生成块，流式生成与手动编辑均在此进行）。
"""
import datetime
import glob
import html
import math
import os

import gradio as gr
import pandas as pd

from backend import LlamaCppBackend, LocalBackend, OpenAICompatBackend
from core import (
    avg_ppl_of,
    blocks_ppls,
    build_prompt,
    parse_doc,
    ppl_color,
    reconcile_active_ppl,
    serialize_doc,
)
from skills import SKILLS_DIR, import_skill_file, scan_skills, skills_to_context

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

# 单用户本地应用：三种后端常驻（切换不卸载已加载模型），按当前模式取用
_BACKENDS = {
    "local": LocalBackend(),
    "llamacpp": LlamaCppBackend(),
    "api": OpenAICompatBackend(),
}
_ACTIVE = {"kind": "local"}


def _backend():
    return _BACKENDS[_ACTIVE["kind"]]


# ==================================================================== 困惑度可视化
# 活动生成单元的逐 token 困惑度：token_texts 拼接恒等于活动块当前文本，
# ppls 中 None 表示该段无数据（手动编辑区域），渲染为灰色。定稿
# （on_add_generate）时随块存入 block["ppl"]，覆盖全部生成块。
_ACTIVE_PPL = {"token_texts": [], "token_ppls": []}


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


# ==================================================================== 事件处理器
def _reasoning_note(backend):
    """思维链能力描述（加载完成状态展示）。"""
    rs = backend.reasoning_support()
    if rs["supported"] == "yes":
        s = "🧠 支持思维链"
    elif rs["supported"] == "no":
        s = "🧠 不支持思维链"
    else:
        s = "🧠 思维链待检测（生成时自动确认）"
    if rs["toggleable"]:
        s += "，可用「思考模式」开关"
    return s


def on_load_model(mode, model_path, base_url, api_key, api_model,
                  llama_path, llama_gpu, llama_ctx):
    """按模式加载：local=本地模型；llamacpp=GGUF 量化模型；api=OpenAI 兼容 API。"""
    backend = _BACKENDS["api"] if mode == "api" else _BACKENDS[mode or "local"]
    if mode == "api":
        yield {model_status: f"⏳ 正在连接 API：{base_url} …"}
        try:
            backend.load(base_url, api_key, api_model)
            _ACTIVE["kind"] = "api"
            yield {
                model_status: (
                    f"✅ API 已连接：{backend.model} @ {backend.base_url}"
                    "｜logprobs（逐 token 困惑度）将在首次生成时自动探测"
                    f"｜{_reasoning_note(backend)}"
                )
            }
        except Exception as e:  # noqa: BLE001
            yield {model_status: f"❌ API 连接失败：{e}"}
        return
    if mode == "llamacpp":
        path = (llama_path or "").strip()
        if not path:
            yield {model_status: "❌ 请先填写 GGUF 模型路径或 HF GGUF 仓库 ID"}
            return
        yield {model_status: f"⏳ 正在加载 GGUF 量化模型：{path} …"}
        try:
            backend.load(path, n_gpu_layers=int(llama_gpu), n_ctx=int(llama_ctx))
            _ACTIVE["kind"] = "llamacpp"
            yield {
                model_status: (
                    f"✅ 已加载：{backend.model_name}"
                    + (f"｜{backend.quant_info}" if backend.quant_info else "")
                    + f"｜设备：{backend.device}｜上下文：{backend.context_size}"
                    "｜逐 token / 上下文困惑度均可用"
                    f"｜{_reasoning_note(backend)}"
                )
            }
        except Exception as e:  # noqa: BLE001
            yield {model_status: f"❌ 加载失败：{e}"}
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
                f"｜{_reasoning_note(backend)}"
            )
        }
    except Exception as e:  # noqa: BLE001
        yield {model_status: f"❌ 加载失败：{e}"}


def on_generate(blocks, active_text, cot_text, enabled_skill_names, skills_list,
                max_new_tokens, do_sample, temperature, top_k, top_p,
                repetition_penalty, context_mode, thinking_mode):
    backend = _backend()
    if not backend.loaded:
        yield {status_tb: "❌ 模型尚未加载/连接，请先在顶栏完成加载"}
        return

    # 思考模式（推理模型）：auto→None（模型默认）、on→True、off→False
    enable_thinking = {"auto": None, "on": True, "off": False}.get(thinking_mode or "auto")

    enabled = [s for s in (skills_list or []) if s.name in (enabled_skill_names or [])]
    skill_ctx = skills_to_context(enabled)
    try:
        prompt = build_prompt(blocks or [], active_text or "", skill_ctx,
                              mode=context_mode or "chat", backend=backend,
                              enable_thinking=enable_thinking)
    except Exception as e:  # 模型无聊天模板等
        yield {status_tb: f"❌ 上下文构造失败（可切换为 prefix/raw 模式）：{e}"}
        return
    if not prompt or (isinstance(prompt, str) and not prompt.strip()):
        yield {status_tb: "❌ 文档为空：请先添加提示词块并输入内容"}
        return

    base = active_text or ""
    cot_base = cot_text or ""
    # 对账：编辑点之前的着色保留，编辑区域及之后合并为无数据段(None)
    seg_t, seg_p = reconcile_active_ppl(
        _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"], base
    )
    _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"] = seg_t, seg_p
    blocks = blocks or []

    n_skills = len(enabled)
    all_ppls = blocks_ppls(blocks, _ACTIVE_PPL["token_ppls"])
    yield {
        status_tb: f"⏳ 正在计算上下文困惑度…（启用技能 {n_skills} 个）",
        avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
        ppl_plot: plot_data(all_ppls),
        heatmap_md: doc_heatmap_html(blocks, base),
    }

    try:
        # 上下文困惑度：本地 transformers / llama.cpp 模式可用（API 不返回
        # prompt token 概率）
        ctx_ppl_val = None
        if backend.kind in ("local", "llamacpp"):
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
            enable_thinking=enable_thinking,
        )
        for upd in backend.generate_stream(prompt, **params):
            full_text = base + upd.cum_text
            # 累积：对账后的基线覆盖段 + 本轮新 token（拼接恒等于 full_text）
            _ACTIVE_PPL["token_texts"] = seg_t + list(upd.token_texts)
            _ACTIVE_PPL["token_ppls"] = seg_p + list(upd.token_ppls)
            all_ppls = blocks_ppls(blocks, _ACTIVE_PPL["token_ppls"])
            cache_note = getattr(backend, "last_cache_info", "")
            yield {
                active_cell_tb: full_text,
                cot_cell_tb: cot_base + (upd.reasoning_cum or ""),
                avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
                ppl_plot: plot_data(all_ppls),
                heatmap_md: doc_heatmap_html(blocks, full_text),
                ctx_ppl_num: ctx_ppl_val,
                status_tb: (
                    f"{'✅ 生成完成' if upd.final else '⏳ 生成中'}…"
                    f"｜本轮 token：{len(upd.token_ppls)}"
                    f"｜累计着色：{len(all_ppls)}｜"
                    f"平均困惑度：{avg_ppl_of(all_ppls) or 0:.2f}"
                    + (
                        f"｜🧠 思维链：{len(upd.reasoning_cum or '')} 字"
                        + (
                            f"｜思维链困惑度：{avg_ppl_of(upd.reasoning_token_ppls) or 0:.2f}"
                            if upd.reasoning_token_ppls
                            else "（API 无数据）"
                        )
                        if upd.reasoning_cum
                        else ""
                    )
                    + (f"｜{cache_note}" if cache_note and backend.kind in ("local", "llamacpp") else "")
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


def on_add_generate(blocks, active_text, cot_text):
    blocks = list(blocks or [])
    c = (active_text or "").strip()
    # 本轮思维链随块定稿：落为 cot 块（<!-- cot ... -->，渲染不可见）
    cot_c = (cot_text or "").strip()
    if cot_c:
        blocks.append({"type": "cot", "content": cot_c})
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
    # 自动锁定：新生成块（前一个生成块）之前的全部块折叠为标题栏，
    # 聚焦当前写作；用户可随时点开标题栏解锁编辑
    for b in blocks[:-1]:
        b.setdefault("locked", True)
    _ACTIVE_PPL["token_texts"], _ACTIVE_PPL["token_ppls"] = [], []
    all_ppls = blocks_ppls(blocks, _ACTIVE_PPL["token_ppls"])
    return {
        blocks_state: blocks,
        active_cell_tb: "",
        cot_cell_tb: "",
        avg_ppl_num: round(avg_ppl_of(all_ppls) or 0.0, 2) if all_ppls else None,
        ppl_plot: plot_data(all_ppls),
        heatmap_md: doc_heatmap_html(blocks, ""),
        status_tb: "已定稿当前生成块并开启新块（前序块已自动锁定折叠，困惑度数据随块保留）",
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


def toggle_block_lock(blocks_list, idx):
    """翻转指定块的锁定状态（锁定=折叠为标题栏且不可编辑）。"""
    blocks_list = [dict(b) for b in (blocks_list or [])]
    if idx < 0 or idx >= len(blocks_list):
        return gr.skip()
    blocks_list[idx]["locked"] = not blocks_list[idx].get("locked", False)
    return blocks_list


# 定时自动保存：仅内容变化时落盘，并轮转仅保留最近 _AUTOSAVE_KEEP 份
_AUTOSAVE_KEEP = 20
_LAST_AUTOSAVE_TEXT = None


def on_autosave(blocks, active_text, enabled):
    if not enabled:
        return {autosave_tb: ""}
    content = serialize_doc(blocks or [], active_text or "")
    if not content.strip():
        return {autosave_tb: ""}
    global _LAST_AUTOSAVE_TEXT
    if content == _LAST_AUTOSAVE_TEXT:
        return {autosave_tb: "🕒 无变化，跳过自动保存"}
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 微秒级时间戳：避免同一秒内多次内容不同的自动保存互相覆盖
    fname = f"autosave_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.md"
    path = os.path.join(SAVE_DIR, fname)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        _LAST_AUTOSAVE_TEXT = content
        files = sorted(glob.glob(os.path.join(SAVE_DIR, "autosave_*.md")),
                       key=os.path.getmtime)
        for old in files[:-_AUTOSAVE_KEEP]:
            os.remove(old)
    except OSError as e:  # 磁盘不可写等：不中断定时器
        return {autosave_tb: f"⚠️ 自动保存失败：{e}"}
    return {autosave_tb: f"🕒 已自动保存 {datetime.datetime.now():%H:%M:%S} → {fname}"}


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
            cot_cell_tb: "",
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


def on_embed_skills(blocks, enabled_skill_names, skills_list):
    """把勾选技能固化为文档顶部的 system 块：文档自包含，
    换到没有该技能的环境也能复现生成过程。"""
    enabled = [s for s in (skills_list or []) if s.name in (enabled_skill_names or [])]
    if not enabled:
        return {status_tb: "❌ 请先勾选要固化的技能"}
    sys_blocks = [{"type": "system", "content": skills_to_context([s])} for s in enabled]
    blocks = sys_blocks + [dict(b) for b in (blocks or [])]
    return {
        blocks_state: blocks,
        skills_check: gr.update(value=[]),  # 已固化，避免重复拼入
        status_tb: (
            f"📥 已将 {len(enabled)} 个技能固化为系统提示词块"
            "（写入文档顶部，保存后自包含，可在无该技能的环境复现生成）"
        ),
    }


# ==================================================================== UI
_initial_skills = scan_skills()

with gr.Blocks(title="生成式文本编辑器") as demo:
    gr.Markdown("# 🖋 生成式文本编辑器\n类 Jupyter 分块：提示词块引导生成，生成块流式续写、可暂停可手动编辑。")

    # 顶栏：后端模式 + 模型加载
    backend_mode_rd = gr.Radio(
        choices=[("🏠 本地模型（transformers）", "local"),
                 ("🦙 llama.cpp（GGUF 量化模型）", "llamacpp"),
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
    with gr.Group(visible=False) as llama_cfg_group:
        gr.Markdown(
            "<span style='font-size:12px;color:#888'>进程内加载 llama.cpp 量化模型"
            "（Q4_K_M / Q5_K_M / Q8_0 / IQ 系列等 GGUF）：小显存/纯 CPU 也能跑大模型。"
            "路径填本地 .gguf 文件（如 models/qwen2.5-0.5b-instruct-q4_k_m.gguf）"
            "或 HF GGUF 仓库 ID（如 Qwen/Qwen2.5-0.5B-Instruct-GGUF，首次自动下载）。"
            "需已安装 llama-cpp-python：pip install llama-cpp-python</span>"
        )
        with gr.Row():
            llama_path_tb = gr.Textbox(
                label="GGUF 模型路径 / HF 仓库 ID",
                value="", placeholder="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                scale=3,
            )
            llama_load_btn = gr.Button("加载 GGUF", variant="primary", scale=1)
        with gr.Row():
            llama_gpu_sl = gr.Slider(
                -1, 100, value=-1, step=1,
                label="n_gpu_layers（GPU offload 层数；-1=全部，0=纯 CPU）",
            )
            llama_ctx_sl = gr.Slider(
                512, 32768, value=4096, step=512,
                label="n_ctx（上下文窗口 / KV cache 容量）",
            )
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
                thinking_mode_rd = gr.Radio(
                    choices=[
                        ("auto（模型默认）", "auto"),
                        ("on（强制思考）", "on"),
                        ("off（关闭思考·提速）", "off"),
                    ],
                    value="auto",
                    label="🧠 思考模式（推理模型：DeepSeek-R1/Qwen3/GLM 等；"
                          "本地=聊天模板 enable_thinking，API=chat_template_kwargs，"
                          "llama.cpp 由 GGUF 模板决定）",
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
                    embed_skills_btn = gr.Button(
                        "📥 固化选中技能为系统块", size="sm"
                    )
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
                    t = blk["type"]
                    if t == "prompt":
                        base_label = f"📝 提示词块 #{i + 1}"
                    elif t == "system":
                        base_label = f"🛠 系统提示词块 #{i + 1}（自包含技能，随文档保存）"
                    elif t == "cot":
                        base_label = f"🧠 思维链块 #{i + 1}（默认隐藏，可编辑）"
                    else:
                        base_label = f"⚙️ 生成块 #{i + 1}"
                    locked = bool(blk.get("locked"))
                    with gr.Group():
                        # 锁定块折叠为可展开标题栏（点击展开查看/解锁）；
                        # 思维链块默认折叠（对应 Markdown 渲染中的隐藏）
                        with gr.Accordion(
                            (f"🔒 {base_label} · 已锁定（点击展开）" if locked else base_label),
                            open=not locked and t != "cot",
                        ):
                            with gr.Row():
                                tb = gr.Textbox(
                                    value=blk["content"],
                                    label=base_label,
                                    lines=4,
                                    interactive=not locked,
                                    scale=20,
                                )
                                del_btn = gr.Button("🗑 删除", size="sm", scale=1)
                            lock_btn = gr.Button(
                                "🔓 解锁（取消折叠）" if locked else "🔒 锁定（折叠隐藏）",
                                size="sm",
                            )

                            def _update(blocks_list, text, idx=i):
                                blocks_list = [dict(b) for b in (blocks_list or [])]
                                if idx >= len(blocks_list) or blocks_list[idx]["content"] == text:
                                    return gr.skip()  # 无变化：跳过，避免触发重渲染
                                blocks_list[idx]["content"] = text
                                blocks_list[idx].pop("ppl", None)  # 内容已改，着色数据失效
                                return blocks_list

                            def _toggle(blocks_list, idx=i):
                                return toggle_block_lock(blocks_list, idx)

                            def _delete(blocks_list, idx=i):
                                return [b for j, b in enumerate(blocks_list or []) if j != idx]

                            # 用 blur 而非 change：change 在 IME 按 Enter 确认候选词时
                            # 也会触发，导致 blocks_state 更新 → @gr.render 销毁重建
                            # 正在输入的组件 → SSE 响应解析失败（Could not parse
                            # server response）。blur 仅失焦时触发，规避该竞态。
                            tb.blur(_update, inputs=[blocks_state, tb], outputs=[blocks_state])
                            lock_btn.click(_toggle, inputs=[blocks_state], outputs=[blocks_state])
                            del_btn.click(_delete, inputs=[blocks_state], outputs=[blocks_state])

            # 当前轮思维链：推理模型（DeepSeek-R1/GLM/Qwen3 等）流式显示于此，
            # 定稿时随块保存为 cot 注释块（Markdown 渲染中不可见）
            with gr.Accordion("🧠 思维链（当前轮·推理模型流式显示，定稿时随块保存）", open=False):
                cot_cell_tb = gr.Textbox(
                    label="思维链（当前轮）— 可手动编辑",
                    lines=5,
                    placeholder="推理模型生成时，思考过程在此流式显示（正文在下方生成块）；非推理模型此处为空",
                    interactive=True,
                )

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

            with gr.Row():
                autosave_cb = gr.Checkbox(
                    value=True, label="定时自动保存（每 60 秒，仅内容变化时落盘）", scale=1
                )
                autosave_tb = gr.Textbox(label="自动保存记录", value="", interactive=False, scale=3)

    # ---------------------------------------------------------------- 事件接线
    _load_inputs = [backend_mode_rd, model_path_tb, api_base_tb, api_key_tb,
                    api_model_tb, llama_path_tb, llama_gpu_sl, llama_ctx_sl]
    load_btn.click(on_load_model, inputs=_load_inputs, outputs=[model_status])
    api_load_btn.click(on_load_model, inputs=_load_inputs, outputs=[model_status])
    llama_load_btn.click(on_load_model, inputs=_load_inputs, outputs=[model_status])

    def _switch_backend_mode(mode):
        return (
            gr.Group(visible=mode == "local"),
            gr.Group(visible=mode == "llamacpp"),
            gr.Group(visible=mode == "api"),
        )

    backend_mode_rd.change(_switch_backend_mode, inputs=[backend_mode_rd],
                           outputs=[local_cfg_group, llama_cfg_group, api_cfg_group])

    generate_btn.click(
        on_generate,
        inputs=[
            blocks_state, active_cell_tb, cot_cell_tb, skills_check, skills_state,
            max_new_tokens_sl, do_sample_cb, temperature_sl, top_k_sl,
            top_p_sl, rep_pen_sl, context_mode_rd, thinking_mode_rd,
        ],
        outputs=[
            active_cell_tb, cot_cell_tb, avg_ppl_num, ppl_plot, heatmap_md,
            ctx_ppl_num, status_tb,
        ],
    )
    stop_btn.click(on_stop, inputs=None, outputs=[status_tb])
    add_prompt_btn.click(on_add_prompt, inputs=[blocks_state],
                         outputs=[blocks_state, status_tb])
    add_generate_btn.click(on_add_generate,
                           inputs=[blocks_state, active_cell_tb, cot_cell_tb],
                           outputs=[blocks_state, active_cell_tb, cot_cell_tb,
                                    avg_ppl_num, ppl_plot, heatmap_md, status_tb])
    save_btn.click(on_save, inputs=[blocks_state, active_cell_tb],
                   outputs=[download_file, status_tb])
    upload_doc_file.change(on_load_doc, inputs=[upload_doc_file],
                           outputs=[blocks_state, active_cell_tb, cot_cell_tb,
                                    avg_ppl_num, ppl_plot, heatmap_md, status_tb])
    refresh_skills_btn.click(on_refresh_skills, inputs=None,
                             outputs=[skills_state, skills_check, skills_info, status_tb])
    embed_skills_btn.click(on_embed_skills,
                           inputs=[blocks_state, skills_check, skills_state],
                           outputs=[blocks_state, skills_check, status_tb])
    upload_skill_file.change(on_upload_skill, inputs=[upload_skill_file],
                             outputs=[skills_state, skills_check, skills_info, status_tb])

    # 定时自动保存：每 60 秒触发一次（后台不可见组件）
    autosave_timer = gr.Timer(60)
    autosave_timer.tick(on_autosave,
                        inputs=[blocks_state, active_cell_tb, autosave_cb],
                        outputs=[autosave_tb])

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), inbrowser=True)

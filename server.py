# -*- coding: utf-8 -*-
"""生成式文本编辑器 — 无头 REST/SSE 服务 + Web 块编辑器托管。

复用 backend.py（本地 transformers / llama.cpp GGUF / OpenAI 兼容三后端）与
core.py（文档序列化、上下文组装、困惑度聚合），默认监听 127.0.0.1:8907。
浏览器打开 http://127.0.0.1:8907/ 即为块编辑器前端（web/，纯 vanilla JS）。

端点：
- GET  /api/status   {kind, loaded, loading, message, generating}
- GET  /api/monitor  status + 模型/显存/内存/生成速率（侧边栏监控面板轮询）
- POST /api/unload   卸载当前后端模型（释放显存/内存）
- POST /api/load     {mode, model_path} 或 {mode, base_url, api_key, model}
                     或 {mode, model_path, n_gpu_layers, n_ctx}（mode=llamacpp）
                     → 立即返回 {accepted:true}，实际加载在后台线程，结果经
                       /api/status 的 loading/message 反映（轮询）
- POST /api/generate {blocks, active_text, skills, params, context_mode}
                     → SSE 流：ctx_ppl / prefill_ppl / update* / error
- POST /api/stop     请求停止当前生成
- GET  /api/skills   [{name, description, instructions}]
- 文档 CRUD（JSON 块数组，存 saves/*.json）：
  GET/POST /api/docs、GET/DELETE /api/docs/{id}、
  POST /api/docs/import_md（旧 Markdown 转 块）、GET /api/docs/{id}/export_md
"""
import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import LlamaCppBackend, LocalBackend, OpenAICompatBackend
from core import (
    _THINK_CLOSE,
    _THINK_OPEN,
    build_prompt,
    normalize_cot,
    parse_doc,
    serialize_doc,
)
from skills import scan_skills, skills_to_context

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8907
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = FastAPI(title="Generative Text Editor Server")

# 三种后端常驻复用（同一时刻仅一个为活动后端）；切换加载时释放其余
# 已加载后端的模型（llama.cpp / transformers 显存即时归还），活动后端
# 重载同模式时旧模型在替换时随之释放
_BACKENDS = {
    "local": LocalBackend(),
    "llamacpp": LlamaCppBackend(),
    "api": OpenAICompatBackend(),
}
_ACTIVE = {"kind": "local"}


def _backend():
    return _BACKENDS[_ACTIVE["kind"]]


# --------------------------------------------------------------- 加载状态
_LOADING = {"running": False, "message": ""}
_GEN_LOCK = threading.Lock()  # 服务级串行化：同一时刻只允许一个生成流

# 当前/最近一次生成的统计（/api/monitor 上报）：tokens 为正文+思维链 token 数
_GEN_STATS = {"tokens": 0, "start": None, "end": None}

# 服务级停止信号：/api/stop 置位，新生成请求清除。
# 后端 _stop_event 在 score_context/prefill 阶段（生成未产出任何 token）会被
# generate_stream 开头的重置吞掉——该阶段点停止必须靠它兜底：_stream() 在
# 进入 generate_stream 前检查，命中则直接结束（锁立即释放，SSE 正常收尾）。
_GEN_ABORT = threading.Event()


class LoadRequest(BaseModel):
    mode: str = "local"
    model_path: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    n_gpu_layers: int = -1   # llamacpp：GPU offload 层数（-1=全部，0=纯 CPU）
    n_ctx: int = 4096        # llamacpp：上下文窗口


class GenParams(BaseModel):
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    # 思考模式（推理模型）：None=模型默认；False=关闭（Qwen3 模板 /
    # vLLM chat_template_kwargs）；True 显式开启
    enable_thinking: Optional[bool] = None


class GenerateRequest(BaseModel):
    blocks: list = Field(default_factory=list)
    active_text: str = ""
    skills: list = Field(default_factory=list)  # 启用的技能名列表
    params: GenParams = Field(default_factory=GenParams)
    context_mode: str = "chat"


@app.get("/api/status")
def status():
    b = _backend()
    return {
        "kind": _ACTIVE["kind"],
        "loaded": b.loaded,
        "loading": _LOADING["running"],
        "message": _LOADING["message"] or ("已加载" if b.loaded else "未加载"),
        "generating": _GEN_LOCK.locked(),
        # 思维链能力：supported=yes/no/unknown、toggleable=可开关
        "reasoning": b.reasoning_support(),
        # 供前端（VSCode 插件）校验服务进程的解释器是否与配置一致
        "python": sys.executable,
        "pid": os.getpid(),
    }


@app.post("/api/load")
def load(req: LoadRequest):
    if _LOADING["running"]:
        return JSONResponse({"error": "已有模型加载任务进行中"}, status_code=409)
    mode = req.mode or "local"
    if mode == "api":
        if not (req.base_url and req.model):
            return JSONResponse({"error": "base_url 与 model 不能为空"}, status_code=400)
    elif mode not in _BACKENDS:
        return JSONResponse(
            {"error": f"未知后端模式：{mode}（可选 local/llamacpp/api）"}, status_code=400
        )
    elif not (req.model_path or "").strip():
        return JSONResponse({"error": "model_path 不能为空"}, status_code=400)

    _LOADING.update(running=True, message="")
    thread = threading.Thread(target=_load_worker, args=(mode, req), daemon=True)
    thread.start()
    return {"accepted": True, "mode": mode}


def _load_worker(mode: str, req: LoadRequest):
    """后台加载线程：完成后把结果写入 _LOADING（status 轮询展示）。"""
    backend = _BACKENDS.get(mode) or _BACKENDS["local"]
    try:
        # 切换后端：先释放其他已加载后端的模型显存/内存，再加载新后端；
        # 生成进行中不得卸载（卸载正在生成的模型会导致硬崩溃）
        if _GEN_LOCK.locked():
            raise RuntimeError("生成进行中，请先停止生成再切换后端")
        for kind, b in _BACKENDS.items():
            if kind != mode and b.loaded:
                try:
                    b.unload()
                except Exception:  # noqa: BLE001  释放尽力而为，不阻断加载
                    traceback.print_exc()
        if mode == "api":
            _LOADING["message"] = f"⏳ 正在连接 API：{req.base_url} …"
            backend.load(req.base_url, req.api_key, req.model)
        elif mode == "llamacpp":
            _LOADING["message"] = (
                f"⏳ 正在加载 GGUF 量化模型：{req.model_path} …"
            )
            backend.load(
                req.model_path,
                n_gpu_layers=req.n_gpu_layers,
                n_ctx=req.n_ctx,
            )
        else:
            _LOADING["message"] = (
                f"⏳ 正在加载模型：{req.model_path} （首次会自动下载，请耐心等待）"
            )
            backend.load(req.model_path)
        _ACTIVE["kind"] = mode
        _LOADING["message"] = (
            f"✅ API 已连接：{backend.model} @ {backend.base_url}" if mode == "api"
            else f"✅ 已加载：{backend.model_name}｜{backend.quant_info}｜"
                 f"设备：{backend.device}｜上下文：{backend.context_size}"
            if mode == "llamacpp"
            else f"✅ 已加载：{backend.model_name}｜设备：{backend.device}"
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()  # 命令行留全堆栈，便于调试
        _LOADING["message"] = f"❌ 加载失败：{e}"
    finally:
        _LOADING["running"] = False


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    backend = _backend()
    if not backend.loaded:
        return JSONResponse({"error": "模型尚未加载，请先加载"}, status_code=400)

    blocks = req.blocks or []
    active = req.active_text or ""
    mode = req.context_mode or "chat"

    # 无活动生成单元（最后一个块不是 generate）时，按最后一块生成块为活动块
    if not blocks or blocks[-1].get("type") != "generate":
        return JSONResponse(
            {"error": "文档缺少活动生成单元：请先创建生成块（<!-- generate -->）"}, status_code=400
        )
    blocks, active = blocks[:-1], blocks[-1].get("content", "")

    try:
        skills = scan_skills()
        enabled = [s for s in skills if s.name in (req.skills or [])]
        skill_ctx = skills_to_context(enabled)
        prompt = build_prompt(blocks, active, skill_ctx, mode=mode, backend=backend,
                              enable_thinking=req.params.enable_thinking)
        if not prompt or (isinstance(prompt, str) and not prompt.strip()):
            return JSONResponse({"error": "文档为空：请先添加提示词块并输入内容"}, status_code=400)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()  # 命令行留全堆栈，便于调试
        return JSONResponse({"error": f"上下文构造失败：{e}"}, status_code=400)

    if not _GEN_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "已有生成任务进行中"}, status_code=409)
    _GEN_ABORT.clear()  # 新一轮生成：清除上一次 /api/stop 的残留信号

    p = req.params
    gen_kwargs = dict(
        max_new_tokens=int(p.max_new_tokens),
        do_sample=bool(p.do_sample),
        temperature=float(p.temperature),
        top_k=int(p.top_k),
        top_p=float(p.top_p),
        repetition_penalty=float(p.repetition_penalty),
        enable_thinking=p.enable_thinking,  # API 后端落为 chat_template_kwargs
    )

    def _stream():
        _GEN_STATS.update(tokens=0, start=None, end=None)
        try:
            # 上下文打分：本地 transformers / llama.cpp 模式可用（API 不可用）。
            # score_context 在同一次前向里顺带产出活动块文本（含手动编辑
            # 部分）的逐 token ppl → prefill_ppl 事件，前端据此为编辑文本着色
            if backend.kind in ("local", "llamacpp"):
                ctx, prefill_t, prefill_p = None, [], []
                try:
                    if hasattr(backend, "score_context"):
                        # 打分阶段也是停止检查点：llamacpp 分块 eval 块间即时
                        # 中止（已 eval 前缀保持为 KV 缓存）；local 单次前向在
                        # 开始前中止（前向本身不可中断，由下方兜底）
                        ctx, prefill_t, prefill_p = backend.score_context(
                            prompt, active, abort_event=_GEN_ABORT
                        )
                    else:
                        ctx = backend.compute_context_ppl(prompt)
                    ctx = round(ctx, 2) if ctx == ctx else None
                except Exception:  # noqa: BLE001
                    ctx, prefill_t, prefill_p = None, [], []

                # 停止请求在 prefill 打分阶段到达（该阶段不响应后端
                # stop_event，且 generate_stream 开头的 _stop_event 重置会
                # 吞掉它）→ 直接收尾，不再下发打分事件
                if _GEN_ABORT.is_set():
                    yield _sse_event("update", {
                        "cum_text": "", "reasoning_cum": "",
                        "token_texts": [], "token_ppls": [],
                        "reasoning_token_texts": [], "reasoning_token_ppls": [],
                        "reasoning_closed": False, "final": True, "cache_info": "",
                    })
                    return
                if ctx is not None:
                    yield _sse_event("ctx_ppl", {"ppl": ctx})
                if prefill_t:
                    yield _sse_event("prefill_ppl", {
                        "token_texts": prefill_t,
                        "token_ppls": prefill_p,
                    })

            prev_tokens = 0  # 上一帧累积 token 数（各后端 token_texts 均为全量快照）
            for upd in backend.generate_stream(prompt, **gen_kwargs):
                if _GEN_ABORT.is_set():
                    # 双保险：停止信号在进入 generate_stream 后（prefill 中）才
                    # 置位时，后端 event 可能已被开头重置吞掉 → 这里兜底收尾
                    yield _sse_event("update", {
                        "cum_text": "", "reasoning_cum": "",
                        "token_texts": [], "token_ppls": [],
                        "reasoning_token_texts": [], "reasoning_token_ppls": [],
                        "reasoning_closed": False, "final": True, "cache_info": "",
                    })
                    return
                if _GEN_STATS["start"] is None:
                    # tps 从首个 token 起算：不含 prefill 打分耗时
                    _GEN_STATS["start"] = time.time()
                # 增量统计：GenUpdate 携带的是从生成开始到当前的累积 token 列表，
                # 直接 += len() 会把 1+2+…+N 全部累加（N≈80 时虚报约 3 千），
                # 必须取本帧与上一帧的差值
                cur_tokens = len(upd.token_texts or []) + len(
                    upd.reasoning_token_texts or []
                )
                _GEN_STATS["tokens"] += cur_tokens - prev_tokens
                prev_tokens = cur_tokens
                cache_note = getattr(backend, "last_cache_info", "")
                yield _sse_event("update", {
                    "cum_text": upd.cum_text,
                    "reasoning_cum": upd.reasoning_cum,  # 思维链（推理模型）
                    "token_texts": upd.token_texts,
                    "token_ppls": upd.token_ppls,
                    # 思维链困惑度（本地/llama.cpp；API 无 reasoning logprobs）
                    "reasoning_token_texts": upd.reasoning_token_texts,
                    "reasoning_token_ppls": upd.reasoning_token_ppls,
                    # 思考区是否闭合（模型输出过闭标签）：插件端据此在 cot 块
                    # 末尾补写闭标签；未闭合则保持"续写思考"状态（所见即所得）
                    "reasoning_closed": upd.reasoning_closed,
                    "final": upd.final,
                    "cache_info": cache_note if backend.kind in ("local", "llamacpp") else "",
                })
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()  # 命令行留全堆栈，便于调试
            yield _sse_event("error", {"error": str(e)})
        finally:
            _GEN_STATS["end"] = time.time()
            _GEN_LOCK.release()

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stop")
def stop():
    _backend().stop()
    _GEN_ABORT.set()
    return {"stopped": True}


# --------------------------------------------------------------- 监控 / 卸载
# /api/monitor 供 VSCode 侧边栏监控面板轮询（约 2s 一次）。显存/内存采集
# 依赖可选库（pynvml / psutil），缺失时对应字段为 null，前端隐藏。

def _gpu_mem_mb():
    """整机 GPU 显存（MB）：pynvml（local/llamacpp 均适用；多卡取第 0 张）。"""
    try:
        import pynvml  # noqa: PLC0415（可选依赖，惰性导入）

        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return info.total // (1024 * 1024), info.used // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None, None


def _torch_mem_mb(backend):
    """transformers 显存（MB）：torch.cuda 统计（仅 local 后端加载后有意义）。"""
    if backend.kind != "local" or not backend.loaded:
        return None, None
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return (
                torch.cuda.memory_allocated() // (1024 * 1024),
                torch.cuda.memory_reserved() // (1024 * 1024),
            )
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _rss_mb():
    """服务进程常驻内存（MB）：psutil 可选依赖。"""
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/monitor")
def monitor():
    b = _backend()
    gpu_total, gpu_used = _gpu_mem_mb()
    torch_alloc, torch_reserved = _torch_mem_mb(b)
    now = time.time()
    elapsed = None
    if _GEN_STATS["start"] is not None:
        elapsed = round(((_GEN_STATS["end"] or now)) - _GEN_STATS["start"], 1)
    tokens = _GEN_STATS["tokens"]
    tps = round(tokens / elapsed, 1) if elapsed and tokens else None
    return {
        "kind": _ACTIVE["kind"],
        "loaded": b.loaded,
        "loading": _LOADING["running"],
        "message": _LOADING["message"] or ("已加载" if b.loaded else "未加载"),
        "generating": _GEN_LOCK.locked(),
        "model": {
            # API 后端模型名在 .model（字符串）；local 的 .model 是 torch 对象，
            # 仅取字符串型回退，避免把模型对象误当名称序列化
            "name": getattr(b, "model_name", "")
            or (b.model if isinstance(getattr(b, "model", None), str) else ""),
            "device": getattr(b, "device", ""),
            "quant": getattr(b, "quant_info", ""),
            "context_size": getattr(b, "context_size", 0) or 0,
            "base_url": getattr(b, "base_url", ""),
        },
        "gpu": {
            "total_mb": gpu_total,
            "used_mb": gpu_used,
            "torch_allocated_mb": torch_alloc,
            "torch_reserved_mb": torch_reserved,
        },
        "rss_mb": _rss_mb(),
        "gen": {"tokens": tokens, "elapsed_s": elapsed, "tps": tps},
        "pid": os.getpid(),
    }


@app.post("/api/unload")
def unload():
    if _GEN_LOCK.locked():
        return JSONResponse({"error": "生成进行中，请先停止再卸载"}, status_code=409)
    b = _backend()
    if not b.loaded:
        return {"unloaded": False}
    b.unload()
    _LOADING["message"] = "已卸载模型（显存/内存已释放）"
    return {"unloaded": True}


@app.get("/api/skills")
def skills():
    # instructions 一并返回：供前端把技能固化为文档内 system 块（文档自包含）
    return [
        {"name": s.name, "description": s.description, "instructions": s.instructions}
        for s in scan_skills()
    ]


# ==================================================================== 文档 CRUD
# JSON 块数组持久化（saves/*.json）。块边界是前端 UI 元素而非文本标记，
# 服务端只存取结构化块；Markdown 仅作导入/导出交换格式。

_BLOCK_TYPES = {"prompt", "system", "cot", "generate"}
_DOC_ID_RE = re.compile(r"[0-9a-zA-Z_-]{1,64}")


class DocBlock(BaseModel):
    type: str
    content: str = ""
    ppl: Optional[dict] = None  # {token_texts: [...], token_ppls: [...]}
    closed: Optional[bool] = None  # cot 专用：思考是否已结束（新格式；旧格式由标签推导）


class SaveDocRequest(BaseModel):
    id: str = ""
    title: str = "未命名文档"
    blocks: list[DocBlock] = Field(default_factory=list)


class ImportMdRequest(BaseModel):
    text: str
    title: str = "导入的文档"


def _sanitize_blocks(blocks: list[DocBlock]) -> list[dict]:
    """校验并规整块列表；末块非 generate 时补空活动块（文档不变式）。

    cot 块归一化为新格式：content=纯思考正文（标签剥离），closed 属性
    显式表达"是否已结束思考"（旧格式由内容里的标签推导）。
    """
    out = []
    for b in blocks or []:
        if b.type not in _BLOCK_TYPES:
            raise ValueError(f"未知块类型：{b.type}")
        ppl = None
        if isinstance(b.ppl, dict) and isinstance(b.ppl.get("token_texts"), list) \
                and isinstance(b.ppl.get("token_ppls"), list):
            ppl = {
                "token_texts": b.ppl["token_texts"],
                "token_ppls": b.ppl["token_ppls"],
            }
        item = {"type": b.type, "content": b.content.strip(), "ppl": ppl}
        if b.type == "cot":
            body, closed = normalize_cot(
                {"content": b.content, "closed": b.closed}
            )
            item["content"] = body
            item["closed"] = closed
        out.append(item)
    if not out or out[-1]["type"] != "generate":
        out.append({"type": "generate", "content": "", "ppl": None})
    return out


def _doc_path(doc_id: str) -> Optional[str]:
    if not _DOC_ID_RE.fullmatch(doc_id or ""):
        return None
    return os.path.join(SAVE_DIR, f"{doc_id}.json")


def _write_doc(doc_id: str, title: str, blocks: list[dict]) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    doc = {
        "id": doc_id,
        "title": title,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "blocks": blocks,
    }
    with open(_doc_path(doc_id), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def _read_doc(doc_id: str) -> Optional[dict]:
    path = _doc_path(doc_id)
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/docs")
def list_docs():
    """文档列表（按更新时间倒序）。"""
    docs = []
    if os.path.isdir(SAVE_DIR):
        for name in os.listdir(SAVE_DIR):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(SAVE_DIR, name), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                docs.append({
                    "id": doc.get("id") or name[:-5],
                    "title": doc.get("title") or "未命名文档",
                    "updated_at": doc.get("updated_at") or "",
                })
            except (OSError, ValueError):
                continue  # 跳过损坏的文档文件
    docs.sort(key=lambda d: d["updated_at"], reverse=True)
    return docs


@app.post("/api/docs")
def save_doc(req: SaveDocRequest):
    try:
        blocks = _sanitize_blocks(req.blocks)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    doc_id = req.id if (_DOC_ID_RE.fullmatch(req.id or "") and _doc_path(req.id)) \
        else uuid.uuid4().hex[:12]
    _write_doc(doc_id, req.title.strip() or "未命名文档", blocks)
    return {"id": doc_id}


@app.get("/api/docs/{doc_id}")
def get_doc(doc_id: str):
    doc = _read_doc(doc_id)
    if doc is None:
        return JSONResponse({"error": "文档不存在"}, status_code=404)
    return doc


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str):
    path = _doc_path(doc_id)
    if not path or not os.path.isfile(path):
        return JSONResponse({"error": "文档不存在"}, status_code=404)
    os.remove(path)
    return {"deleted": True}


@app.post("/api/docs/import_md")
def import_md(req: ImportMdRequest):
    """旧 Markdown 注释块格式 → JSON 块文档（一次性按需迁移）。"""
    blocks, active = parse_doc(req.text or "")
    if active or not blocks:
        blocks = blocks + [{"type": "generate", "content": active, "ppl": None}]
    blocks = _sanitize_blocks([DocBlock(**b) for b in blocks])
    doc_id = uuid.uuid4().hex[:12]
    title = (req.title or "").strip() or "导入的文档"
    _write_doc(doc_id, title, blocks)
    return {"id": doc_id, "title": title, "blocks": blocks}


@app.get("/api/docs/{doc_id}/export_md")
def export_md(doc_id: str):
    """JSON 块文档 → Markdown（core.serialize_doc，与旧格式互通）。"""
    doc = _read_doc(doc_id)
    if doc is None:
        return JSONResponse({"error": "文档不存在"}, status_code=404)
    blocks = _sanitize_blocks([DocBlock(**b) for b in doc.get("blocks", [])])
    active = blocks.pop()["content"]  # 末块 = 活动生成单元
    # cot 块补回标签（md 交换格式以标签承载"是否已结束思考"，与旧格式互通）
    for b in blocks:
        if b["type"] == "cot":
            b["content"] = _THINK_OPEN + "\n" + b["content"] + (
                "\n" + _THINK_CLOSE if b.get("closed") else ""
            )
    text = serialize_doc(
        [{"type": b["type"], "content": b["content"]} for b in blocks], active
    )
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


# 静态托管 Web 块编辑器（挂在最后，不遮挡 /api/* 路由）
# no-cache：ES module 会被浏览器启发式缓存，改动后旧 JS 不刷新（依赖
# ETag 重验证，未变时仍为 304，开销可忽略）
if os.path.isdir(WEB_DIR):
    _static = StaticFiles(directory=WEB_DIR, html=True)

    @app.middleware("http")
    async def _static_no_cache(request, call_next):
        resp = await call_next(request)
        if not request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/", _static, name="web")


def main():
    parser = argparse.ArgumentParser(description="生成式文本编辑器无头服务")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

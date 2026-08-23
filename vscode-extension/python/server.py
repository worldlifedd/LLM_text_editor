# -*- coding: utf-8 -*-
"""生成式文本编辑器 — 无头 REST/SSE 服务（供 VSCode 插件等前端使用）。

复用 backend.py（本地 transformers / OpenAI 兼容双后端）与 core.py
（文档序列化、上下文组装、困惑度聚合），默认监听 127.0.0.1:8907。

端点：
- GET  /api/status   {kind, loaded, loading, message, generating}
- POST /api/load     {mode, model_path} 或 {mode, base_url, api_key, model}
                     → 立即返回 {accepted:true}，实际加载在后台线程，结果经
                       /api/status 的 loading/message 反映（轮询）
- POST /api/generate {blocks, active_text, skills, params, context_mode}
                     → SSE 流：ctx_ppl / update* / error
- POST /api/stop     请求停止当前生成
- GET  /api/skills   [{name, description}]
"""
import argparse
import json
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import LocalBackend, OpenAICompatBackend
from core import build_prompt
from skills import scan_skills, skills_to_context

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8907

app = FastAPI(title="Generative Text Editor Server")

# 单用户本地应用：两种后端常驻（切换不卸载本地模型），按当前模式取用
_BACKENDS = {"local": LocalBackend(), "api": OpenAICompatBackend()}
_ACTIVE = {"kind": "local"}


def _backend():
    return _BACKENDS[_ACTIVE["kind"]]


# --------------------------------------------------------------- 加载状态
_LOADING = {"running": False, "message": ""}
_GEN_LOCK = threading.Lock()  # 服务级串行化：同一时刻只允许一个生成流


class LoadRequest(BaseModel):
    mode: str = "local"
    model_path: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class GenParams(BaseModel):
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1


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
    }


@app.post("/api/load")
def load(req: LoadRequest):
    if _LOADING["running"]:
        return JSONResponse({"error": "已有模型加载任务进行中"}, status_code=409)
    mode = req.mode or "local"
    if mode == "api":
        if not (req.base_url and req.model):
            return JSONResponse({"error": "base_url 与 model 不能为空"}, status_code=400)
    elif not (req.model_path or "").strip():
        return JSONResponse({"error": "model_path 不能为空"}, status_code=400)

    _LOADING.update(running=True, message="")
    thread = threading.Thread(target=_load_worker, args=(mode, req), daemon=True)
    thread.start()
    return {"accepted": True, "mode": mode}


def _load_worker(mode: str, req: LoadRequest):
    """后台加载线程：完成后把结果写入 _LOADING（status 轮询展示）。"""
    backend = _BACKENDS["api"] if mode == "api" else _BACKENDS["local"]
    try:
        _LOADING["message"] = (
            f"⏳ 正在连接 API：{req.base_url} …" if mode == "api"
            else f"⏳ 正在加载模型：{req.model_path} （首次会自动下载，请耐心等待）"
        )
        if mode == "api":
            backend.load(req.base_url, req.api_key, req.model)
        else:
            backend.load(req.model_path)
        _ACTIVE["kind"] = mode
        _LOADING["message"] = (
            f"✅ API 已连接：{backend.model} @ {backend.base_url}" if mode == "api"
            else f"✅ 已加载：{backend.model_name}｜设备：{backend.device}"
        )
    except Exception as e:  # noqa: BLE001
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
            {"error": "文档缺少活动生成单元：请先创建 <generate> 块"}, status_code=400
        )
    blocks, active = blocks[:-1], blocks[-1].get("content", "")

    try:
        skills = scan_skills()
        enabled = [s for s in skills if s.name in (req.skills or [])]
        skill_ctx = skills_to_context(enabled)
        prompt = build_prompt(blocks, active, skill_ctx, mode=mode, backend=backend)
        if not prompt or (isinstance(prompt, str) and not prompt.strip()):
            return JSONResponse({"error": "文档为空：请先添加提示词块并输入内容"}, status_code=400)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"上下文构造失败：{e}"}, status_code=400)

    if not _GEN_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "已有生成任务进行中"}, status_code=409)

    p = req.params
    gen_kwargs = dict(
        max_new_tokens=int(p.max_new_tokens),
        do_sample=bool(p.do_sample),
        temperature=float(p.temperature),
        top_k=int(p.top_k),
        top_p=float(p.top_p),
        repetition_penalty=float(p.repetition_penalty),
    )

    def _stream():
        try:
            # 上下文困惑度仅本地模式可用
            if backend.kind == "local":
                try:
                    ctx = backend.compute_context_ppl(prompt)
                    ctx = round(ctx, 2) if ctx == ctx else None
                except Exception:  # noqa: BLE001
                    ctx = None
                if ctx is not None:
                    yield _sse_event("ctx_ppl", {"ppl": ctx})

            for upd in backend.generate_stream(prompt, **gen_kwargs):
                cache_note = getattr(backend, "last_cache_info", "")
                yield _sse_event("update", {
                    "cum_text": upd.cum_text,
                    "token_texts": upd.token_texts,
                    "token_ppls": upd.token_ppls,
                    "final": upd.final,
                    "cache_info": cache_note if backend.kind == "local" else "",
                })
        except Exception as e:  # noqa: BLE001
            yield _sse_event("error", {"error": str(e)})
        finally:
            _GEN_LOCK.release()

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stop")
def stop():
    _backend().stop()
    return {"stopped": True}


@app.get("/api/skills")
def skills():
    return [{"name": s.name, "description": s.description} for s in scan_skills()]


def main():
    parser = argparse.ArgumentParser(description="生成式文本编辑器无头服务")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

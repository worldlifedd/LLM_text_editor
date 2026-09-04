# Mock OpenAI 兼容服务（带思维链）：用于验证 VSCode 插件「思维链块（cot）」全链路。
#
# 与 tests/_mock_openai.py 的区别：本服务会先流式吐 reasoning_content（思考过程），
# 再流式吐 content（正文，带 logprobs），从而覆盖 cot 块的创建 / 流式写入 /
# 定稿冻结 / 困惑度着色路径。
#
# 用法：
#   C:/Users/worldlife/miniconda3/envs/gte/python.exe tests/mock_reasoning_openai.py
#
# 插件侧（GTE 生成控制面板 → API 模式）：
#   base_url = http://127.0.0.1:8998/v1
#   api_key  = 任意（如 mock）
#   model    = mock-r1-reasoning      ← 名字含 r1/reason，面板直接显示「✅ 支持思维链」
#
# 模型名说明：API 后端 reasoning_support() 走模型名启发式，命中
# backend._REASONING_NAME_HINTS（r1/qwq/reason/thinking/gpt-o/glm-z/glm4-z）
# 才会静态判定为「支持」；否则显示 ❓，直到首次收到 reasoning_content 转 ✅。
# 想测「❓ → ✅ 运行时探测」，把 model 改成 mock-plain 即可。
#
# 控制台会打印每次请求的关键信息：messages（可核对 cot 块未回灌）、
# enable_thinking（思考模式开关是否送达）、logprobs/chat_template_kwargs。

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

MODEL = "mock-r1-reasoning"

# 思考过程（reasoning_content）：逐段吐出，便于观察 cot 块流式增长
REASONING = [
    "嗯，用户要一段关于秋天的散文开头。",
    "先想想意象：梧桐、青石板、斜阳、桂香，这几个都带秋意。",
    "节奏上短句起手更有画面感，后面再铺长句。",
    "字数控制在 100 字左右，不要过度铺陈。",
    "好，就从这个方向写。",
]

# 正文（content）
ANSWER = [
    "秋日的午后，", "阳光穿过梧桐叶", "洒在青石板上，",
    "碎成一地斑驳的", "金色。", "风过处，", "桂香浮动，",
    "像是谁把整个夏天", "轻轻合上了。",
]

# 正文逐 token 对数概率：前段确定（绿），末段困惑（红），用于验证色阶
LOGPROBS = [-0.05, -0.12, -0.30, -0.55, -1.10, -1.90, -2.60, -3.20, -3.60]


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL}]}


def _dump(tag, obj):
    """控制台诊断输出（中文不转义）。"""
    print(f"  {tag}: {json.dumps(obj, ensure_ascii=False)}", flush=True)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    max_tokens = int(body.get("max_tokens", 256))
    use_logprobs = "logprobs" in body

    print("\n" + "=" * 60, flush=True)
    print("POST /v1/chat/completions", flush=True)
    _dump("model", body.get("model"))
    _dump("max_tokens", max_tokens)
    _dump("logprobs", use_logprobs)
    # 思考模式开关：server.py 在 enable_thinking=False 时发 chat_template_kwargs
    _dump("chat_template_kwargs", body.get("chat_template_kwargs"))
    # 核对上下文组装：cot 块不应出现在 messages 里（推理模型自行重新思考）
    for i, m in enumerate(body.get("messages") or []):
        c = (m.get("content") or "").replace("\n", "\\n")
        print(f"  messages[{i}] {m.get('role'):9s} {c[:90]}", flush=True)

    def gen():
        # ---- 第一段：思维链（reasoning_content）----
        for seg in REASONING:
            chunk = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": seg},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            time.sleep(0.06)

        # ---- 第二段：正文（content + logprobs）----
        n = min(len(ANSWER), max_tokens)
        for i, w in enumerate(ANSWER[:n]):
            chunk = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}],
            }
            if use_logprobs:
                chunk["choices"][0]["logprobs"] = {
                    "content": [{"token": w, "logprob": LOGPROBS[i],
                                 "top_logprobs": []}]
                }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            time.sleep(0.05)

        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print(f"Mock reasoning OpenAI server on http://127.0.0.1:8998/v1  (model={MODEL})",
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8998)

# 临时 Mock OpenAI 兼容服务：验证 server.py 的 API 后端 generate SSE 链路。
# 监听 127.0.0.1:8999/v1/chat/completions，流式返回若干 chunk（含 logprobs）。
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

WORDS = ["秋", "日", "的", "午", "后", "，", "阳", "光", "洒", "落", "。"]


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "mock-model"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    max_tokens = body.get("max_tokens", 16)
    use_logprobs = "logprobs" in body

    def gen():
        for i, w in enumerate(WORDS[:max_tokens]):
            lp = -0.1 if i < len(WORDS) - 3 else -3.0
            chunk = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}],
            }
            if use_logprobs:
                chunk["choices"][0]["logprobs"] = {
                    "content": [{"token": w, "logprob": lp, "top_logprobs": []}]
                }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            time.sleep(0.05)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8998)

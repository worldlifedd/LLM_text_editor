# -*- coding: utf-8 -*-
"""服务端 SSE 集成测试（TestClient + 替身后端）：

验证 /api/generate 的事件流契约：
- ctx_ppl：上下文困惑度
- prefill_ppl：活动块文本（含手动编辑部分）的逐 token ppl（新增事件）
- update / error / final 快照
替身后端不打分误差，专注事件编排与字段透传。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from backend import GenUpdate


class _FakeBackend:
    kind = "llamacpp"
    loaded = True
    model_name = "fake-model"
    quant_info = "Q4_K_M"
    device = "CPU"
    context_size = 512
    last_cache_info = "⚡KV缓存复用 3/5 token"

    def reasoning_support(self):
        return {"supported": "no", "toggleable": False}

    def build_chat_prompt(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        body = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return f"HEAD{body}{cot_prefix}{active_text}"

    def build_flat_prompt(self, flat_text):
        return flat_text

    def score_context(self, context, tail_text="", abort_event=None):
        # tail 恰为 context 后缀时产出逐 token ppl；否则（对齐失败）仅 ppl
        if tail_text and context.endswith(tail_text):
            return 12.34, list(tail_text), [float(ord(c)) % 7 + 1 for c in tail_text]
        return 12.34, [], []

    def generate_stream(self, context, **gen_params):
        yield GenUpdate(
            cum_text="起了", token_texts=["起", "了"], token_ppls=[3.5, 4.5],
        )
        yield GenUpdate(
            cum_text="起了。", token_texts=["起", "了", "。"], token_ppls=[3.5, 4.5, 5.5],
            final=True,
        )

    def stop(self):
        pass


server._BACKENDS["llamacpp"] = _FakeBackend()
server._ACTIVE["kind"] = "llamacpp"

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)


def _events(payload):
    """POST /api/generate → [(event, data)] 列表。"""
    out = []
    with client.stream("POST", "/api/generate", json=payload) as r:
        assert r.status_code == 200, r.status_code
        ev, buf = None, ""
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                buf += line[5:].strip()
            elif line == "" and ev is not None:
                out.append((ev, json.loads(buf)))
                ev, buf = None, ""
    return out


REQ = {
    # 服务端契约：blocks 尾部的 generate 块即活动生成单元（其 content 为
    # 活动文本，active_text 字段仅作冗余备份）
    "blocks": [
        {"type": "prompt", "content": "写散文"},
        {"type": "generate", "content": "秋风"},
    ],
    "active_text": "秋风",
    "skills": [],
    "params": {},
    "context_mode": "chat",
}

# 1. 事件编排：ctx_ppl → prefill_ppl → update*（含 final）
evs = _events(REQ)
names = [e for e, _ in evs]
assert names[0] == "ctx_ppl", names
assert names[1] == "prefill_ppl", names
assert names[-1] == "update" and evs[-1][1]["final"] is True, names
ctx_ev = dict(evs)["ctx_ppl"]
assert ctx_ev == {"ppl": 12.34}, ctx_ev
pre_ev = dict(evs)["prefill_ppl"]
assert pre_ev["token_texts"] == ["秋", "风"], pre_ev
assert len(pre_ev["token_ppls"]) == 2, pre_ev
upd = evs[-1][1]
assert upd["cum_text"] == "起了。" and upd["token_ppls"] == [3.5, 4.5, 5.5], upd
assert upd["cache_info"].startswith("⚡"), upd
print("[1] SSE 事件编排（ctx_ppl → prefill_ppl → update/final）OK:", names)

# 2. 无活动文本（空 generate 块）→ 仅 ctx_ppl，无 prefill_ppl 事件
evs = _events({**REQ, "blocks": REQ["blocks"][:-1] + [{"type": "generate", "content": ""}],
               "active_text": ""})
names = [e for e, _ in evs]
assert "prefill_ppl" not in names and names[0] == "ctx_ppl", names
print("[2] 空活动文本 → 无 prefill_ppl 事件 OK:", names)

# 3. score_context 异常 → 降级为无事件，生成流不受影响
orig = _FakeBackend.score_context


def _boom(self, context, tail_text="", abort_event=None):
    raise RuntimeError("scoring failed")


_FakeBackend.score_context = _boom
try:
    evs = _events(REQ)
    names = [e for e, _ in evs]
    assert "ctx_ppl" not in names and "prefill_ppl" not in names, names
    assert names[-1] == "update" and evs[-1][1]["final"] is True, names
finally:
    _FakeBackend.score_context = orig
print("[3] score_context 异常降级 OK")

# 4. 临近思维链恒回灌：正文已开始时 prompt 仍含思考区且位于正文之前——
#    续写上下文与首次生成一致 → prefill 打分不漂移、KV 前缀可整体复用
#    （修复前：正文一旦开始 CoT 即被整体丢弃，已生成文字着色随续写变化）
_seen = {}
_orig_score = _FakeBackend.score_context


def _rec_score(self, context, tail_text="", abort_event=None):
    _seen["context"] = context
    return _orig_score(self, context, tail_text, abort_event=abort_event)


_FakeBackend.score_context = _rec_score
try:
    evs = _events({**REQ, "blocks": [
        {"type": "prompt", "content": "写散文"},
        {"type": "cot", "content": "构思", "closed": True},
        {"type": "generate", "content": "秋风"},
    ]})
    ctx = _seen["context"]
    _O, _C = "<" + "thi" + "nk>", "</" + "thi" + "nk>"
    assert _O + "\n构思\n" + _C in ctx, ctx
    assert ctx.endswith("秋风") and ctx.index("构思") < ctx.index("秋风"), ctx
    assert "prefill_ppl" in [e for e, _ in evs], [e for e, _ in evs]
finally:
    _FakeBackend.score_context = _orig_score
print("[4] 正文已开始仍回灌临近思维链 OK")

print("ALL SERVER PREFILL TESTS PASSED")

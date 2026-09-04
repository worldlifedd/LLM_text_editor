# -*- coding: utf-8 -*-
"""三后端统一契约测试：同一套 GenUpdate 不变式跑 local / llamacpp / api。

后端公共接口（backend.py 模块头所列方法）之外，真正的行为契约是
GenUpdate 流——前端（web/editor.js _renderContent 的前缀覆盖判定）与
VSCode 插件（generation.ts reconcileActivePpl 公共前缀对账）都按下述
不变式着色，任一后端违反即整块不着色：

  C1 末个更新 final=True 且唯一，其余 final=False
  C2 token_texts / token_ppls 一一等长（reasoning 同理）
  C3 join(token_texts) 恒为 cum_text 的前缀；final 时相等（logprobs 可用）
  C4 join(reasoning_token_texts) 恒为 reasoning_cum 的前缀；final 时相等，
     且不含思考区标签字符（仅思考正文）；API 后端无 reasoning logprobs
     → 恒为空（思维链不着色是已知降级，非失配）

历史教训（本文件的回归锚点，防后端行为再度不一致）：
- llamacpp：flush 尾部未补进 token 序列 → join < reasoning_cum（已修）
- local：思考区标签字符混进 reasoning_token_texts → join 含 写实
  标签，与 reasoning_cum 失配 → 思维链不着色（[L2]/[L3] 回归）
- local：think 态不消费 token，思考未闭合（截断/停止）→ 思维链永不
  着色（[L4] 回归）

用法：python tests/test_backend_contract.py（local 用例需 venv 含 torch）
"""
import ctypes
import inspect
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import backend
from backend import GenUpdate, LlamaCppBackend, LocalBackend, OpenAICompatBackend

O = "<" + "thi" + "nk>"
C = "</" + "thi" + "nk>"
N_VOCAB = 64
# 单点高置信分布（logit=5，其余 0）下选中 token 的困惑度
PPL_ONEHOT = 1.0 + (N_VOCAB - 1) * math.exp(-5.0)


# ============================================================ 契约校验器
def check_contract(updates, *, label, token_ppl=True, reasoning_ppl=True):
    """GenUpdate 流的统一不变式（C1–C4）。"""
    assert updates, f"{label}: 无更新"
    finals = [u for u in updates if u.final]
    assert finals == [updates[-1]], f"{label} C1: final 快照必须且只能是最后一个"

    for u in updates:
        assert len(u.token_texts) == len(u.token_ppls), \
            f"{label} C2: token 文本/ppl 长度失配 {len(u.token_texts)}≠{len(u.token_ppls)}"
        assert len(u.reasoning_token_texts) == len(u.reasoning_token_ppls), \
            f"{label} C2: reasoning 文本/ppl 长度失配"
        cov = "".join(u.token_texts)
        if token_ppl:
            assert u.cum_text.startswith(cov), \
                f"{label} C3: token 覆盖非 cum_text 前缀: {cov!r} ⊄ {u.cum_text!r}"
        rcov = "".join(u.reasoning_token_texts)
        if reasoning_ppl:
            assert u.reasoning_cum.startswith(rcov), \
                f"{label} C4: reasoning 覆盖非 reasoning_cum 前缀: {rcov!r} ⊄ {u.reasoning_cum!r}"
        else:
            assert not rcov, \
                f"{label} C4: 该后端不应产出 reasoning ppl: {rcov!r}"

    fin = updates[-1]
    if token_ppl:
        assert "".join(fin.token_texts) == fin.cum_text, \
            f"{label} C3(final): {fin.token_texts!r} != {fin.cum_text!r}"
    if reasoning_ppl:
        assert "".join(fin.reasoning_token_texts) == fin.reasoning_cum, \
            f"{label} C4(final): reasoning 覆盖 {fin.reasoning_token_texts!r} != {fin.reasoning_cum!r}"


# ============================================================ 统一接口
COMMON_METHODS = [
    "loaded", "load", "unload", "kind", "stop",
    "build_chat_prompt", "build_flat_prompt",
    "compute_context_ppl", "generate_stream", "reasoning_support",
]
for cls in (LocalBackend, LlamaCppBackend, OpenAICompatBackend):
    for m in COMMON_METHODS:
        assert hasattr(cls, m), f"统一接口缺失: {cls.__name__}.{m}"
assert len({inspect.signature(c.build_chat_prompt) for c in
            (LocalBackend, LlamaCppBackend, OpenAICompatBackend)}) == 1, \
    "build_chat_prompt 签名必须三后端一致"
print("[0] 统一接口 + build_chat_prompt 签名一致 OK")

# [0b] 统一接口行为：flat 文本构造 + API 侧 chat prompt（cot → reasoning_content
#     回灌，续写上下文与首次生成一致；local/llamacpp 的模板路径由
#     test_server_prefill.py [4] 经服务端链路覆盖）
_fl, _fg, _fa = LocalBackend(), LlamaCppBackend(), OpenAICompatBackend()
assert _fl.build_flat_prompt("平文本") == "平文本"
assert _fg.build_flat_prompt("平文本") == "平文本"
assert _fa.build_flat_prompt("平文本") == [{"role": "user", "content": "平文本"}]
_m = _fa.build_chat_prompt([{"role": "user", "content": "写散文"}], "正文", cot_prefix="思考")
assert _m[-1] == {"role": "assistant", "content": "正文", "reasoning_content": "思考"}, _m[-1]
_m2 = _fa.build_chat_prompt([], "", cot_prefix="思考")
assert _m2 == [{"role": "assistant", "content": "", "reasoning_content": "思考"}], _m2
print("[0b] 统一接口行为：flat 构造 / API cot 回灌 OK")


# ============================================================ local（transformers）
class _FakeTok:
    """字符级 tokenizer 替身：__call__/decode 与真实接口签名一致。"""

    def __init__(self, char2id, vocab):
        self.char2id, self.vocab = char2id, vocab

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [self.char2id[c] for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


class _FakeHFModel:
    """最小 HF 模型替身：脚本化逐 token 生成，复刻 generate 的关键调用序
    （先 streamer.put(prompt) 首跳；每步先过 logits_processor / 停止判据、
    再选中 token 并 put；分布单点高置信在将选 token 上）。

    走 backend._local_deps() 的真实 _TokenStream / _PplProcessor /
    _StopOnEvent / DynamicCache（真实 torch/transformers 5.x 语义），只把
    模型前向与采样替身化——对齐逻辑因此是被真实流式管线驱动的。
    """

    def __init__(self, gen_ids):
        import torch
        self.torch = torch
        self.config = types.SimpleNamespace()  # 无 layer_types → 非混合注意力
        self.device = "cpu"
        self.gen_ids = gen_ids

    def __call__(self, **kw):
        return None  # prefill 前向（KV 缓存内容测试不校验）

    def generate(self, input_ids=None, attention_mask=None, streamer=None,
                 logits_processor=None, stopping_criteria=None,
                 max_new_tokens=256, **kw):
        torch = self.torch
        streamer.put(input_ids)  # generate 先 put 整个 prompt（skip_prompt 首跳）
        cur = input_ids
        for n, t in enumerate(self.gen_ids):
            if n >= max_new_tokens:
                break
            scores = torch.zeros((1, N_VOCAB))
            scores[0, t] = 5.0
            for p in logits_processor:
                scores = p(cur, scores)
            if stopping_criteria(cur, scores):
                break
            cur = torch.cat([cur, torch.tensor([[t]])], dim=1)
            streamer.put(torch.tensor([[t]]))
        streamer.end()


def _local_backend(gen_text, context="写散文"):
    chars = []
    for ch in gen_text + context:
        if ch not in chars:
            chars.append(ch)
    vocab = {i + 2: ch for i, ch in enumerate(chars)}
    vocab[0] = ""
    char2id = {ch: i for i, ch in vocab.items() if ch}
    be = LocalBackend()
    be.tokenizer = _FakeTok(char2id, vocab)
    be.model = _FakeHFModel([char2id[c] for c in gen_text])
    be.model_name = "fake-local"
    return be


# [L1] 无思维链纯正文：正文着色、契约成立
be = _local_backend("秋风起了")
ups = list(be.generate_stream("写散文", max_new_tokens=8, do_sample=False))
check_contract(ups, label="[L1]")
assert ups[-1].cum_text == "秋风起了" and ups[-1].reasoning_cum == ""
assert all(abs(p - PPL_ONEHOT) < 1e-5 for p in ups[-1].token_ppls)
print("[L1] local 纯正文 OK:", ups[-1].cum_text)

# [L2] 开标签在生成流内（DeepSeek-R1 / Qwen3 默认形态）——回归重点：
#     修复前标签字符混进 reasoning_token_texts，join != reasoning_cum，
#     思维链整块不着色（正文因无标签污染而正常着色）
be = _local_backend(O + "\n思考" + C + "\n正文")
ups = list(be.generate_stream("写散文", max_new_tokens=32, do_sample=False))
check_contract(ups, label="[L2]")
assert ups[-1].reasoning_cum == "\n思考", ups[-1].reasoning_cum
assert ups[-1].cum_text == "\n正文", ups[-1].cum_text
assert ups[-1].reasoning_closed
assert all(abs(p - PPL_ONEHOT) < 1e-5 for p in ups[-1].reasoning_token_ppls)
print("[L2] local 流内开标签（思维链着色回归）OK:",
      repr(ups[-1].reasoning_cum), "->", len(ups[-1].reasoning_token_ppls), "tokens")

# [L3] 开标签已在 prompt 内（Qwen3 enable_thinking=true 模板形态）
be = _local_backend("思考内容" + C + "\n答案")
ups = list(be.generate_stream("写散文" + O + "\n", max_new_tokens=32, do_sample=False))
check_contract(ups, label="[L3]")
assert ups[-1].reasoning_cum == "思考内容" and ups[-1].cum_text == "\n答案"
assert ups[-1].reasoning_closed
print("[L3] local prompt 内开标签 OK:", repr(ups[-1].reasoning_cum))

# [L4] 思考未闭合（流自然结束/截断，无闭标签）——回归：修复前 think 态
#     不消费 token，思维链永远拿不到 ppl；现在流式期间即着色，结束后也完整
be = _local_backend(O + "\n想了一半还没")
ups = list(be.generate_stream("写散文", max_new_tokens=32, do_sample=False))
check_contract(ups, label="[L4]")
assert ups[-1].reasoning_cum == "\n想了一半还没", ups[-1].reasoning_cum
assert ups[-1].cum_text == "" and not ups[-1].reasoning_closed
print("[L4] local 思考未闭合截断（着色兜底）OK:", repr(ups[-1].reasoning_cum))


# ============================================================ llamacpp
class _FakeLlama:
    """最小 Llama 替身（同 test_llamacpp_mock 语义）：生成序列逐 token
    yield，yield 前布置该 token 的单点高置信 logits。"""

    def __init__(self):
        self.ctx = self
        self._logits = np.zeros(N_VOCAB, dtype=np.float32)
        self._logits_ptr = self._logits.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        self._model = types.SimpleNamespace(vocab=None)
        self.metadata = {"general.name": "fake", "tokenizer.chat_template": "T"}
        self._input_ids = []
        self.n_tokens = 0
        self.gen_seq = []

    def n_ctx(self):
        return 512

    def n_vocab(self):
        return N_VOCAB

    def token_bos(self):
        return 0

    def token_eos(self):
        return 1

    def tokenize(self, b, special=True):
        return [self._c2i[c] for c in b.decode("utf-8")]

    def detokenize(self, ids, prev_tokens=None):
        return "".join(self._i2c[i] for i in ids).encode("utf-8")

    def reset(self):
        pass

    def eval(self, ids):
        self._set_onehot(ids[-1])

    def _set_onehot(self, t):
        self._logits[:] = 0.0
        self._logits[t] = 5.0

    def generate(self, tokens, top_k=50, top_p=0.95, temp=0.8, repeat_penalty=1.1):
        self._input_ids = list(tokens)
        self.n_tokens = len(tokens)
        for t in self.gen_seq:
            self._set_onehot(t)
            yield t


def _llamacpp_backend(gen_text, context="写散文"):
    chars = []
    for ch in gen_text + context:
        if ch not in chars:
            chars.append(ch)
    fake = _FakeLlama()
    fake._i2c = {i + 2: ch for i, ch in enumerate(chars)}
    fake._i2c[0] = ""
    fake._c2i = {ch: i for i, ch in fake._i2c.items() if ch}
    fake.gen_seq = [fake._c2i[c] for c in gen_text] + [1]  # EOG 收尾
    be = LlamaCppBackend()
    be._llm = fake
    be.model_name = "fake.gguf"
    be.context_size = 512
    return be


# [G1] 无思维链
be = _llamacpp_backend("秋风起了")
ups = list(be.generate_stream("写散文", max_new_tokens=8, do_sample=False))
check_contract(ups, label="[G1]")
assert ups[-1].cum_text == "秋风起了" and ups[-1].reasoning_cum == ""
print("[G1] llamacpp 纯正文 OK:", ups[-1].cum_text)

# [G2] 流内开标签 + 闭合 + 正文
be = _llamacpp_backend(O + "\n思考" + C + "\n正文")
ups = list(be.generate_stream("写散文", max_new_tokens=32, do_sample=False))
check_contract(ups, label="[G2]")
assert ups[-1].reasoning_cum == "\n思考" and ups[-1].cum_text == "\n正文"
assert ups[-1].reasoning_closed
print("[G2] llamacpp 完整思考块 OK:", repr(ups[-1].reasoning_cum))

# [G3] prompt 内开标签 + 思考未闭合（flush 兜底回归）
be = _llamacpp_backend("想了一半还没", "写散文" + O + "\n")
ups = list(be.generate_stream("写散文" + O + "\n", max_new_tokens=32, do_sample=False))
check_contract(ups, label="[G3]")
assert ups[-1].reasoning_cum == "想了一半还没" and ups[-1].cum_text == ""
assert not ups[-1].reasoning_closed
print("[G3] llamacpp 未闭合思考 OK:", repr(ups[-1].reasoning_cum))


# ============================================================ api（OpenAI 兼容）
class _FakeResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        pass


class _FakeRequests:
    """替身 requests 模块：get 校验连接、post 回放脚本化 SSE 行。"""

    RequestException = type("RequestException", (Exception,), {})

    def __init__(self, lines):
        self._lines = lines

    def get(self, url, headers=None, timeout=None):
        return types.SimpleNamespace(status_code=200)

    def post(self, url, headers=None, json=None, stream=True, timeout=None):
        return _FakeResp(self._lines)


def _sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False)


def _api_lines(reasoning, content):
    lines = []
    for piece in reasoning:
        lines.append(_sse({"choices": [{"index": 0, "delta": {"reasoning_content": piece}}]}))
    for piece in content:
        lines.append(_sse({
            "choices": [{
                "index": 0,
                "delta": {"content": piece},
                "logprobs": {"content": [{"token": piece, "logprob": -0.5}]},
            }],
        }))
    lines.append("data: [DONE]")
    return lines


_saved_requests = backend.requests
try:
    backend.requests = _FakeRequests(_api_lines("思考", "正文"))
    be = OpenAICompatBackend()
    be.load("http://fake/v1", "k", "mock-model")
    ups = list(be.generate_stream(
        [{"role": "user", "content": "写散文"}], max_new_tokens=8, do_sample=False
    ))
finally:
    backend.requests = _saved_requests
# API 契约：content 有 logprobs；reasoning 无（协议不回传）→ 恒空
check_contract(ups, label="[A1]", reasoning_ppl=False)
assert ups[-1].reasoning_cum == "思考" and ups[-1].cum_text == "正文"
assert ups[-1].reasoning_closed  # reasoning_content 为协议级字段，天然闭合
assert not ups[-1].reasoning_token_texts
assert all(abs(p - math.exp(0.5)) < 1e-6 for p in ups[-1].token_ppls)
print("[A1] api reasoning_content + content logprobs OK:",
      repr(ups[-1].reasoning_cum), repr(ups[-1].cum_text))

print("\nALL BACKEND CONTRACT TESTS PASSED")

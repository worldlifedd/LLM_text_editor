# -*- coding: utf-8 -*-
"""LlamaCppBackend mock 单测：以替身 Llama 类注入 backend._llama_cpp，
不依赖真实 GGUF 模型与 GPU，验证：
- load：本地路径加载、元数据（量化/上下文/设备）、空路径报错
- llama-cpp-python 未安装时的 ImportError 提示
- 聊天模板：渲染 + active_text 续写头；无模板时报错并提示 prefix/raw
- generate_stream：token-文本对齐、逐 token 困惑度数值、final 快照、
  生成参数映射（do_sample→temp=0、repeat_penalty）、EOG 提前结束、
  优雅停止、max_new_tokens 截断、超长 prompt 尾部保留
- compute_context_ppl：数值正确性、eval 轨迹、空文本 NaN
- KV 缓存注记：未命中 / 前缀复用
- 新版 llama-cpp-python 的 n_ctx 方法形态兼容
"""
import ctypes
import math
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import backend
from backend import LlamaCppBackend

BOS, EOS = 1, 2
# 简化词表：id -> 文本片段
VOCAB = {
    BOS: "<s>", EOS: "</s>",
    10: "秋", 11: "风", 12: "起", 13: "了", 14: "，",
    20: "写", 21: "散", 22: "文",
}
CHAR2ID = {v: k for k, v in VOCAB.items() if k not in (BOS, EOS)}
N_VOCAB = 32
# 单点高置信分布的期望困惑度：1 + (N-1)·e^(-5)
PPL_ONEHOT = 1.0 + (N_VOCAB - 1) * math.exp(-5.0)


# ---------------------------------------------------------------- 替身模型
class FakeLlama:
    """最小 llama_cpp.Llama 替身：可控 logits 缓冲、固定生成序列。

    语义约定（与被测代码的读取时机对齐）：
    - eval([t]) 后的 logits 分布：单点高置信落在 t 本身（预测「重复 t」），
      故打分文本用重复字符即可得到 PPL_ONEHOT。
    - generate() 在 yield token t 前布置 t 的单点分布 → 生成侧逐 token
      困惑度恒为 PPL_ONEHOT。
    """

    def __init__(self, model_path="", n_ctx=512, n_gpu_layers=-1, verbose=False):
        self.model_path = model_path
        self.n_ctx = n_ctx          # 普通属性（旧版形态）；用例可换为方法
        self._logits = np.zeros(N_VOCAB, dtype=np.float32)
        self._logits_ptr = self._logits.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        self.ctx = self
        self._model = types.SimpleNamespace(vocab=None)
        self.metadata = {
            "general.name": "fake-model",
            "general.file_type": 15,  # LLAMA_FTYPE_MOSTLY_Q4_K_M
            "tokenizer.chat_template": "FAKE_TEMPLATE",
        }
        self._input_ids = []
        self.n_tokens = 0
        self.eval_seq = []      # compute_context_ppl 的逐 token eval 轨迹
        self.gen_calls = []     # generate() 收到的调用参数
        self.gen_seq = [10, 11, 12, 14, EOS]  # 默认生成序列（EOG 收尾）

    def n_vocab(self):          # 新版形态：方法
        return N_VOCAB

    # ---- 词表接口 ----
    def token_bos(self):
        return BOS

    def token_eos(self):
        return EOS

    def tokenize(self, b, special=True):
        return [CHAR2ID.get(ch, 13) for ch in b.decode("utf-8")]

    def detokenize(self, ids, prev_tokens=None):
        return "".join(VOCAB.get(i, "?") for i in ids).encode("utf-8")

    # ---- 前向接口 ----
    def reset(self):
        self.eval_seq = []

    def eval(self, ids):
        self.eval_seq.extend(ids)
        self._set_onehot(ids[-1])

    def _set_onehot(self, t):
        self._logits[:] = 0.0
        self._logits[t] = 5.0

    def generate(self, tokens, top_k=50, top_p=0.95, temp=0.8, repeat_penalty=1.1):
        self.gen_calls.append(dict(
            tokens=list(tokens), top_k=top_k, top_p=top_p,
            temp=temp, repeat_penalty=repeat_penalty,
        ))
        self._input_ids = list(tokens)
        self.n_tokens = len(tokens)
        for t in self.gen_seq:
            self._set_onehot(t)  # 产出 token 前布置其原始分布
            yield t


def _make_fake_llama_cpp():
    """构造伪 llama_cpp 模块（含 llama_get_logits / 聊天模板等接口）。"""
    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama

    def llama_get_logits(ctx):
        return ctx._logits_ptr

    def llama_vocab_is_eog(vocab, token):
        return token == EOS

    mod.llama_get_logits = llama_get_logits
    mod.llama_vocab_is_eog = llama_vocab_is_eog
    mod.LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    mod.LLAMA_FTYPE_MOSTLY_Q8_0 = 7

    class _Formatter:
        def __init__(self, template="", eos_token="", bos_token=""):
            self.template = template

        def __call__(self, messages):
            prompt = "".join(
                f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages
            ) + "<assistant>"
            return types.SimpleNamespace(prompt=prompt)

    chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    chat_format.Jinja2ChatFormatter = _Formatter
    mod.llama_chat_format = chat_format
    return mod


def _fresh_backend():
    """注入伪 llama_cpp 后新建后端（互不污染）。"""
    backend._llama_cpp = _make_fake_llama_cpp()
    return LlamaCppBackend(), backend._llama_cpp.Llama


def _load_fake(be, Llama, **kw):
    """经临时文件路径走真实 load() 流程，返回替身实例。"""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
    try:
        be.load(path, **kw)
    finally:
        os.unlink(path)
    return be._llm


# ==================================================================== 用例
# 1. load：本地路径 + 元数据
be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
assert be.loaded and isinstance(fake, FakeLlama)
assert be.model_name and be.context_size == 1024 and be.device == "CPU"
assert be.quant_info == "fake-model｜Q4_K_M", be.quant_info
print("[1] load + metadata OK:", be.quant_info, be.device, be.context_size)

# n_gpu_layers 形态
be2, Llama2 = _fresh_backend()
_load_fake(be2, Llama2, n_gpu_layers=-1, n_ctx=512)
assert be2.device == "GPU(全部层)", be2.device
be3, Llama3 = _fresh_backend()
_load_fake(be3, Llama3, n_gpu_layers=5, n_ctx=512)
assert be3.device == "GPU×5层", be3.device
print("[1b] n_gpu_layers device mapping OK")

# 空路径
be4, _ = _fresh_backend()
try:
    be4.load("  ")
    raise AssertionError("空路径应报错")
except ValueError as e:
    assert "GGUF" in str(e)
print("[1c] empty path ValueError OK")

# 2. 未安装 llama-cpp-python：ImportError 带安装指引
class _BlockLlamaCpp:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "llama_cpp":
            raise ImportError(f"blocked: {name}")
        return None


_saved, backend._llama_cpp = backend._llama_cpp, None
sys.meta_path.insert(0, _BlockLlamaCpp())
try:
    try:
        LlamaCppBackend().load("x.gguf")
        raise AssertionError("应抛 ImportError")
    except ImportError as e:
        assert "pip install llama-cpp-python" in str(e), str(e)
finally:
    sys.meta_path.pop(0)
    backend._llama_cpp = _saved
print("[2] missing-deps ImportError OK")

# 3. 聊天模板
msgs = [
    {"role": "system", "content": "你是作家"},
    {"role": "user", "content": "写秋天"},
]
p = be.build_chat_prompt(msgs, "秋风")
assert p == "<system>你是作家</system><user>写秋天</user><assistant>秋风", p
assert be.build_flat_prompt("平文本") == "平文本"
# 无模板 → 报错并提示 prefix/raw
fake.metadata.pop("tokenizer.chat_template")
try:
    be.apply_chat_template(msgs, "")
    raise AssertionError("无模板应报错")
except ValueError as e:
    assert "prefix/raw" in str(e)
print("[3] chat template render / no-template error OK")

# 4. generate_stream：对齐 + ppl 数值 + final + 参数映射
be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
updates = list(be.generate_stream(
    "写散文", max_new_tokens=64, do_sample=False, temperature=0.7,
    top_k=7, top_p=0.9, repetition_penalty=1.15,
))
final = updates[-1]
assert final.final and not updates[0].final
assert final.cum_text == "秋风起，", final.cum_text
assert "".join(final.token_texts) == final.cum_text, "对齐拼接不一致"
assert len(final.token_ppls) == 4
assert all(abs(p - PPL_ONEHOT) < 1e-6 for p in final.token_ppls), final.token_ppls
# 参数映射：贪心 → temp=0；repeat_penalty 透传
call = fake.gen_calls[-1]
assert call["temp"] == 0.0 and call["top_k"] == 7 and call["top_p"] == 0.9
assert call["repeat_penalty"] == 1.15
# 首次生成：无历史上下文 → 未命中
assert be.last_cache_info == "KV缓存未命中", be.last_cache_info
print("[4] generate_stream align/ppl/params/cache OK:",
      final.cum_text, [round(p, 3) for p in final.token_ppls])

# 4b. do_sample=True → temperature 透传
list(be.generate_stream("写散文", max_new_tokens=2, do_sample=True, temperature=0.55))
assert fake.gen_calls[-1]["temp"] == 0.55
print("[4b] do_sample=True temp passthrough OK")

# 5. KV 前缀缓存注记：同长 prompt 二次生成 → 复用（≥16 token 公共前缀）
long_prompt = "秋" * 20
list(be.generate_stream(long_prompt, max_new_tokens=1, do_sample=False))
assert be.last_cache_info == "KV缓存未命中"  # 首次
# 替身 generate 已把 _input_ids 留为上次 prompt → 本次同 prompt 命中
list(be.generate_stream(long_prompt, max_new_tokens=1, do_sample=False))
assert "KV缓存复用 20/20" in be.last_cache_info, be.last_cache_info
print("[5] cache reuse note OK:", be.last_cache_info)

# 6. max_new_tokens 截断
be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
fake.gen_seq = [10] * 50
updates = list(be.generate_stream("写", max_new_tokens=3, do_sample=False))
assert len(updates[-1].token_ppls) == 3
print("[6] max_new_tokens truncation OK")

# 7. EOG 提前结束：EOS 之后的 token 不产出
fake.gen_seq = [10, EOS, 11, 12]
final = list(be.generate_stream("写", max_new_tokens=10, do_sample=False))[-1]
assert final.cum_text == "秋" and len(final.token_ppls) == 1
print("[7] EOG early-stop OK")

# 8. 优雅停止：首个更新后置位
fake.gen_seq = [10] * 100
stopped_early = False
final = None
for upd in be.generate_stream("写", max_new_tokens=100, do_sample=False):
    if not stopped_early:
        stopped_early = True
        be.stop()
    final = upd
assert final.final and len(final.token_ppls) <= 3, len(final.token_ppls)
print("[8] graceful stop OK: stopped at", len(final.token_ppls), "tokens")

# 9. 未加载即生成 → RuntimeError
be9, _ = _fresh_backend()
try:
    list(be9.generate_stream("x"))
    raise AssertionError("未加载应报错")
except RuntimeError as e:
    assert "加载" in str(e)
print("[9] not-loaded RuntimeError OK")

# 10. compute_context_ppl：重复字符文本 → 每步预测命中 → PPL_ONEHOT
be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
ppl = be.compute_context_ppl("秋秋秋秋")
assert len(fake.eval_seq) == 3, fake.eval_seq  # len(toks)-1 次单 token eval
assert abs(ppl - PPL_ONEHOT) < 1e-6, ppl
assert math.isnan(be.compute_context_ppl("   "))  # 空文本
print("[10] compute_context_ppl OK:", round(ppl, 4))

# 10b. score_context：同趟产出整段 ppl + 尾部（活动块文本）逐 token ppl
ctx, tt, tp = be.score_context("秋秋秋秋", "秋秋")
assert abs(ctx - PPL_ONEHOT) < 1e-6, ctx
assert tt == ["秋", "秋"], tt  # detokenize 差分对齐
assert len(tp) == 2 and all(abs(p - PPL_ONEHOT) < 1e-6 for p in tp), tp
# 尾部非 context 后缀 → 放弃（空列表）
_, tt2, tp2 = be.score_context("秋秋秋秋", "风")
assert tt2 == [] and tp2 == []
# 空尾部 / 空文本
_, tt3, _ = be.score_context("秋秋秋秋", "")
assert tt3 == []
_ctx9, tt9, tp9 = be.score_context("  ", "秋")
assert math.isnan(_ctx9) and tt9 == [] and tp9 == []
# 尾部 == 整个文本（首 token 无前置分布）→ 放弃
_, tt4, _ = be.score_context("秋秋秋秋", "秋秋秋秋")
assert tt4 == []
# 尾部跨 token 边界错位（头部 tokenize 非全量前缀）不会在此词表出现，
# 以 mock 单字 token 特性覆盖等价场景：头部 2 token 边界精确对齐
_, tt5, tp5 = be.score_context("秋风秋风", "秋风")
assert tt5 == ["秋", "风"] and len(tp5) == 2, (tt5, tp5)
print("[10b] score_context tail ppl OK:", tt, [round(p, 3) for p in tp])

# 11. 超长 prompt：保留尾部（最新上下文，n_ctx-8 截断；_n_ctx 有 512 下限）
fake.n_ctx = 520
fake.gen_seq = [10, 11]
list(be.generate_stream("写散文" + "秋" * 516, max_new_tokens=5, do_sample=False))
sent = fake.gen_calls[-1]["tokens"]
assert len(sent) == 512 and set(sent) == {10}, sent[:8]  # 519 token → 保留尾部 512
print("[11] long-prompt tail truncation OK: len =", len(sent))

# 12. n_ctx 方法形态（新版 llama-cpp-python）兼容
fake.n_ctx = lambda: 999
assert be._n_ctx() == 999
print("[12] n_ctx attr/callable compat OK")

# 13. 思维链流式分离：开标签在生成流内（DeepSeek-R1 / Qwen3 默认形态）
# 扩展词表以支持 <think> / <<arg_key:6124c78e>> 的逐字符 detokenize
for _i, _ch in enumerate("<>/think"):
    VOCAB[23 + _i] = _ch     # N_VOCAB=32，23..30 空闲
    CHAR2ID[_ch] = 23 + _i
O = "<" + "thi" + "nk>"
C = "</" + "thi" + "nk>"


def _ids(s):
    return [CHAR2ID[c] for c in s]


be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
fake.gen_seq = _ids(O + "秋风" + C + "起了") + [EOS]
final = list(be.generate_stream("写散文", max_new_tokens=64, do_sample=False))[-1]
assert final.reasoning_cum == "秋风", final.reasoning_cum
assert final.cum_text == "起了", final.cum_text
assert "".join(final.reasoning_token_texts) == final.reasoning_cum
print("[13] 思维链分离（开标签在流内）OK:", repr(final.reasoning_cum), repr(final.cum_text))

# 14. 开标签已在 prompt 内（Qwen3 enable_thinking=true 渲染出未闭合的 <think>）：
#     流首即思考正文且未闭合 → 全靠 flush 把暂缓的尾部补回。
#     这是"思维链不着色"的回归点：若 flush 的 r 没同步进 token 序列，
#     join(reasoning_token_texts) 会比 reasoning_cum 短，插件端对账失配。
be, Llama = _fresh_backend()
fake = _load_fake(be, Llama, n_gpu_layers=0, n_ctx=1024)
fake.gen_seq = _ids("秋风起了，") + [EOS]
final = list(be.generate_stream("写散文" + O + "\n", max_new_tokens=64, do_sample=False))[-1]
assert final.reasoning_cum == "秋风起了，", final.reasoning_cum
assert final.cum_text == "", final.cum_text
assert "".join(final.reasoning_token_texts) == final.reasoning_cum, (
    "".join(final.reasoning_token_texts), final.reasoning_cum,
)
assert len(final.reasoning_token_ppls) == len(final.reasoning_token_texts)
print("[14] 思维链分离 + flush 补尾（开标签在 prompt 内）OK:",
      repr(final.reasoning_cum), "| token 段数", len(final.reasoning_token_texts))

print("ALL LLAMACPP MOCK TESTS PASSED")

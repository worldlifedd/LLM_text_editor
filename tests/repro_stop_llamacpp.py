"""llamacpp 后端 stop 及时性回归测试（替身 llama_cpp 模块，不依赖真实模型）。

覆盖：生成中调用 backend.stop() → 循环须在下一个 token 间隔内退出并 yield
final 快照；随后 generate_stream 结束（锁释放前提）。
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend as backend_mod

STOP_PROPAGATE_SEC = 3.0  # 停止后允许的收尾时间


class FakeLlama:
    def __init__(self, **kw):
        self.ctx = None
        self._model = types.SimpleNamespace(vocab=None)
        self.metadata = {}
        self._input_ids = [1, 2, 3]
        self.n_tokens = 3

    def n_ctx(self):
        return 4096

    def n_vocab(self):
        return 32000

    def tokenize(self, text, special=True):
        return [10] * max(1, len(text) // 3)

    def detokenize(self, ids, prev_tokens=None):
        return b"".join(b"t" for _ in ids)

    def token_eos(self):
        return 2

    def token_bos(self):
        return 1

    def longest_token_prefix(self, a, b):
        return 0

    def close(self):
        pass

    def generate(self, tokens, **kwargs):
        n = 0
        while n < 10000:
            n += 1
            yield 99
            time.sleep(0.15)
        # 永不 natural 结束：只能靠 stop()


def _install_fake_llama_cpp():
    fake = types.ModuleType("llama_cpp")
    fake.Llama = FakeLlama
    fake.llama_get_logits = lambda ctx: None
    fake.llama_vocab_is_eog = lambda vocab, token: False
    fake.llama_chat_format = types.SimpleNamespace(Jinja2ChatFormatter=object)
    # _quant_desc 遍历 _llama_cpp 及其 llama_cpp 子模块属性 → 需可 dir()
    fake.__dict__.setdefault("llama_cpp", fake)
    sys.modules["llama_cpp"] = fake


def test_llamacpp_stop_propagates():
    _install_fake_llama_cpp()
    b = backend_mod.LlamaCppBackend()
    b._llm = FakeLlama()
    b.model_name = "fake.gguf"
    b.context_size = 4096

    gen = b.generate_stream("你好", max_new_tokens=200, do_sample=True)

    first = next(gen)
    assert first.cum_text, "生成应已产出内容"
    t0 = time.time()
    b.stop()
    got_final = False
    for upd in gen:
        if getattr(upd, "final", False):
            got_final = True
            break
    elapsed = time.time() - t0
    assert got_final, "stop 后 generate_stream 未 yield final 快照（未退出）"
    assert elapsed < STOP_PROPAGATE_SEC, f"stop 传播过慢：{elapsed:.1f}s"


def test_llamacpp_score_abort():
    """prefill 打分阶段停止：abort_event 命中须跳过 eval 立即返回不可用。"""
    _install_fake_llama_cpp()
    b = backend_mod.LlamaCppBackend()
    b._llm = FakeLlama()
    b.model_name = "fake.gguf"
    b.context_size = 4096

    ev = threading.Event()
    ev.set()  # 模拟停止请求已到达
    t0 = time.time()
    ctx, tail_t, tail_p = b.score_context("你好" * 200, abort_event=ev)
    assert ctx != ctx, "abort 后应返回 nan"
    assert tail_t == [] and tail_p == []
    assert time.time() - t0 < 1.0, "abort 命中后不应做任何 eval"


if __name__ == "__main__":
    test_llamacpp_stop_propagates()
    test_llamacpp_score_abort()
    print("PASS: llamacpp stop 及时退出 / prefill abort 即时生效")

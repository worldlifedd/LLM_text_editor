"""transformers（local）后端 stop 及时性回归测试（替身依赖，不加载真实模型）。

模拟：generate 线程逐 token 产出，受 stopping_criteria（_StopOnEvent）控制；
stop() 后须在下一个 token 间隔内退出并 yield final 快照。
"""
import contextlib
import os
import queue
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend as backend_mod

STOP_PROPAGATE_SEC = 3.0


class _FakeTensor:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]

    def reshape(self, *a):
        return self

    def tolist(self):
        return list(self.data)

    def item(self):
        return self.data[0]


class _FakeTorch:
    long = "long"

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def tensor(data, dtype=None, device=None):
        return _FakeTensor(data)

    @staticmethod
    def ones(*shape, dtype=None, device=None):
        return _FakeTensor([1] * shape[0])

    @staticmethod
    def arange(*a, device=None):
        return _FakeTensor(list(range(*a)))

    @staticmethod
    def log_softmax(scores, dim=-1):
        return _FakeTensor(0.0)


class _TokenStream:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.token_ids = []
        self._first_put = True
        self.text_queue = queue.Queue()

    def put(self, value):
        if self._first_put:
            self._first_put = False  # generate 先 put(prompt_ids)
        else:
            self.token_ids.append(123)
        self.text_queue.put(self.tokenizer.decode([123], skip_special_tokens=True))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        while True:
            item = self.text_queue.get()
            if item is None:
                break
            yield item


class _PplProcessor:
    def __init__(self):
        self.log_probs = []
        self.prev_logprobs = None


class _StopOnEvent:
    def __init__(self, ev):
        self.ev = ev

    def __call__(self, input_ids, scores, **kwargs):
        return bool(self.ev.is_set())


class _CriteriaList(list):
    def __call__(self, input_ids, scores, **kwargs):
        return any(c(input_ids, scores, **kwargs) for c in self)


class _DynamicCache:
    def crop(self, n):
        pass


class _FakeModel:
    config = types.SimpleNamespace(layer_types=[])
    device = "cpu"

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None,
                 use_cache=True, cache_position=None):
        return types.SimpleNamespace()

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        max_new = int(kwargs["max_new_tokens"])
        for _ in range(max_new):
            if criteria(None, None):  # _StopOnEvent 查 stop_event
                break
            streamer.put(_FakeTensor([123]))
            time.sleep(0.1)
        streamer.end()


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids, skip_special_tokens=True):
        return "t" * len(ids)


def test_local_stop_propagates():
    fake = {
        "torch": _FakeTorch,
        "AutoModelForCausalLM": object,
        "AutoTokenizer": object,
        "DynamicCache": _DynamicCache,
        "LogitsProcessorList": list,
        "StoppingCriteriaList": _CriteriaList,
        "_PplProcessor": _PplProcessor,
        "_TokenStream": _TokenStream,
        "_StopOnEvent": _StopOnEvent,
    }
    backend_mod._LOCAL_DEPS = fake

    b = backend_mod.LocalBackend()
    b.model = _FakeModel()
    b.tokenizer = _FakeTokenizer()

    gen = b.generate_stream("你好", max_new_tokens=500, do_sample=True)
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


def test_local_score_abort():
    """prefill 打分阶段停止：abort_event 命中须跳过前向立即返回不可用。"""
    fake = {
        "torch": _FakeTorch,
        "AutoModelForCausalLM": object,
        "AutoTokenizer": object,
    }
    backend_mod._LOCAL_DEPS = fake

    b = backend_mod.LocalBackend()
    b.model = _FakeModel()
    b.tokenizer = _FakeTokenizer()

    ev = threading.Event()
    ev.set()  # 模拟停止请求已到达
    t0 = time.time()
    ctx, tail_t, tail_p = b.score_context("你好" * 200, abort_event=ev)
    assert ctx != ctx, "abort 后应返回 nan"
    assert tail_t == [] and tail_p == []
    assert time.time() - t0 < 1.0, "abort 命中后不应做任何前向"


if __name__ == "__main__":
    test_local_stop_propagates()
    test_local_score_abort()
    print("PASS: local(transformers) stop 及时退出 / prefill abort 即时生效")

# -*- coding: utf-8 -*-
"""KV 前缀缓存回归：命中加速 / 贪心一致性 / 中途编辑 / 停止后再生成。"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import LocalBackend

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GEN = dict(max_new_tokens=24, do_sample=False, repetition_penalty=1.1)

b = LocalBackend()
print("[0] 加载模型 ...")
t0 = time.perf_counter()
b.load(MODEL)
print(f"    {time.perf_counter() - t0:.0f}s 加载完成")

doc = "秋日的午后，阳光穿过梧桐叶洒在青石板上。" * 12
ctx1 = b.apply_chat_template([{"role": "user", "content": "写一段关于秋天的散文。"}], doc)


def run(ctx, tag):
    """跑一次生成，返回 (首token延迟ms, final快照)。"""
    t = time.perf_counter()
    first_ms, final = None, None
    for upd in b.generate_stream(ctx, **GEN):
        if first_ms is None:
            first_ms = (time.perf_counter() - t) * 1000
        final = upd
    print(f"    [{tag}] 首 token {first_ms:.0f}ms｜{b.last_cache_info}")
    return first_ms, final


print("[1] 基线：无缓存全量 prefill")
b._kv_cache = None
_, u1 = run(ctx1, "基线")
assert len(u1.cum_text) > 5

print("[2] 续写：ctx2 = ctx1 + 生成文本 → 缓存命中 + 贪心输出一致")
ctx2 = ctx1 + u1.cum_text
b._kv_cache = None
ms_full, u_full = run(ctx2, "无缓存对照")
ms_hit, u_hit = run(ctx2, "缓存命中")
assert "复用" in b.last_cache_info, b.last_cache_info
assert u_hit.cum_text == u_full.cum_text, "缓存路径贪心输出与全量不一致！"
print(f"    加速比 {ms_full / ms_hit:.1f}x ✔ 输出一致 ✔")

print("[3] 中途编辑：改动文档中段 → 部分复用")
ctx3 = ctx1[:200] + "忽然，一阵风吹过，" + ctx1[200:] + u1.cum_text
ms_edit, u_edit = run(ctx3, "部分复用")
assert "复用" in b.last_cache_info
assert len(u_edit.cum_text) > 5

print("[4] 停止后缓存一致性（上次失败场景）")
ctx4 = ctx3 + u_edit.cum_text
gen = b.generate_stream(ctx4, max_new_tokens=400, do_sample=False,
                        repetition_penalty=1.1)
holder = {}


def consume():
    try:
        for upd in gen:
            holder["final"] = upd
    except Exception as e:  # noqa: BLE001
        holder["err"] = e


th = threading.Thread(target=consume, daemon=True)
th.start()
time.sleep(1.0)  # 生成进行中
b.stop()
th.join(timeout=30)
assert not th.is_alive(), "停止后消费线程未退出"
assert "err" not in holder, f"生成器异常：{holder.get('err')}"
n_stopped = len(holder["final"].token_ppls)
assert n_stopped < 400, "应提前停止"
print(f"    停止成功（{n_stopped} token）｜{b.last_cache_info}")

print("[5] 停止后立即再生成同上下文 → 线程串行化 + 缓存复用")
ms5, u5 = run(ctx4, "停止后再生成")
assert "复用" in b.last_cache_info, b.last_cache_info
assert len(u5.cum_text) > 5
assert len(u5.token_ppls) == len(u5.token_texts) > 0
assert "".join(u5.token_texts) == u5.cum_text, "ppl 文本对齐失败"
print("    ppl 对齐 ✔")

print("ALL KV CACHE TESTS PASSED ✔")

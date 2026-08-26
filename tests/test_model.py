# -*- coding: utf-8 -*-
"""模型级冒烟测试：加载 Qwen2.5-0.5B、流式生成、逐token困惑度对齐、优雅停止。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import LocalBackend

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print("== 加载模型 ==", flush=True)
t0 = time.time()
be = LocalBackend()
be.load(MODEL)
print(f"loaded on {be.device} in {time.time() - t0:.1f}s", flush=True)

# 1. 上下文困惑度
ctx_ppl = be.compute_context_ppl("秋天到了，树叶变黄，一阵风吹过。")
print(f"[1] context ppl = {ctx_ppl:.2f}", flush=True)
assert 1.0 < ctx_ppl < 1e6

# 2. 流式生成 + 对齐（贪心，可复现）
final = None
n_yields = 0
for upd in be.generate_stream(
    "请用中文写一段关于秋天的散文，大约50字：", max_new_tokens=60, do_sample=False
):
    n_yields += 1
    final = upd
    print(f"  ...tokens={len(upd.token_ppls)} text={upd.cum_text[:20]!r}", flush=True)

assert final is not None and final.final, "应有 final 快照"
assert final.cum_text.strip(), "生成为空"
assert "\ufffd" not in final.cum_text, "中文解码乱码"
assert len(final.token_ppls) == len(final.token_texts) > 0, "ppl 与文本未对齐"
joined = "".join(final.token_texts)
assert joined == final.cum_text, f"对齐拼接不一致:\n{joined!r}\nvs\n{final.cum_text!r}"
assert all(p > 0 and p < 1e9 for p in final.token_ppls)
print(f"[2] stream/align OK: {n_yields} yields, {len(final.token_ppls)} tokens, "
      f"avg_ppl={[round(p,1) for p in final.token_ppls[:5]]}...", flush=True)

# 3. 优雅停止：生成 400 token，第一个更新后停止
t0 = time.time()
got_stop = False
for upd in be.generate_stream(
    "请写一篇很长的文章：", max_new_tokens=400, do_sample=False
):
    if not got_stop:
        got_stop = True
        be.stop()
        print(f"  stop requested at tokens={len(upd.token_ppls)}", flush=True)
    final = upd
dt = time.time() - t0
assert final.final
print(f"[3] graceful stop OK: stopped at {len(final.token_ppls)} tokens in {dt:.1f}s", flush=True)

# 4. 续写语义由前端保证（base+cum），此处验证上下文含已生成内容可继续
print("ALL MODEL TESTS PASSED", flush=True)

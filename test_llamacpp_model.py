# -*- coding: utf-8 -*-
"""LlamaCppBackend 真实模型回归：流式生成、逐token困惑度、上下文困惑度、
优雅停止、KV 缓存复用、聊天模板行为。

默认模型 stories260K.gguf（llama.cpp 官方 260K 参数玩具模型，~1MB，
base 模型无聊天模板）；可用环境变量 GGUF_MODEL 指定任意本地 .gguf
（推荐 Qwen2.5-0.5B-Instruct Q4_K_M 等带聊天模板的 Instruct 量化模型）。
模型文件缺失时跳过（exit 0），便于无模型环境跑 CI。
"""
import math
import os
import sys
import time

from backend import LlamaCppBackend

MODEL = os.environ.get("GGUF_MODEL", "/tmp/models/stories260K.gguf")

if not os.path.exists(MODEL):
    print(f"[skip] 模型不存在：{MODEL}（可设 GGUF_MODEL 指定本地 .gguf）")
    sys.exit(0)

print("== 加载 GGUF 模型 ==", flush=True)
t0 = time.time()
be = LlamaCppBackend()
be.load(MODEL, n_gpu_layers=0, n_ctx=512)
has_template = bool(be._meta().get("tokenizer.chat_template"))
print(
    f"loaded in {time.time() - t0:.1f}s | model={be.model_name} | "
    f"quant={be.quant_info or 'N/A'} | device={be.device} | "
    f"n_ctx={be.context_size} | chat_template={has_template}",
    flush=True,
)
assert be.loaded and be.context_size >= 512

# 1. 上下文困惑度（英文小故事语料，匹配 stories260K 训练分布）
ctx_ppl = be.compute_context_ppl("Once upon a time, there was a little girl.")
print(f"[1] context ppl = {ctx_ppl:.2f}", flush=True)
assert 1.0 < ctx_ppl < 1e6 and ctx_ppl == ctx_ppl, ctx_ppl

# 2. 流式生成 + 对齐（贪心，可复现）
final = None
n_yields = 0
for upd in be.generate_stream(
    "Once upon a time", max_new_tokens=48, do_sample=False
):
    n_yields += 1
    final = upd
assert final is not None and final.final, "应有 final 快照"
assert final.cum_text.strip(), "生成为空"
assert "\ufffd" not in final.cum_text, "解码乱码"
assert len(final.token_ppls) == len(final.token_texts) > 0, "ppl 与文本未对齐"
assert "".join(final.token_texts) == final.cum_text, "对齐拼接不一致"
assert all(0 < p < 1e9 for p in final.token_ppls)
print(
    f"[2] stream/align OK: {n_yields} yields, {len(final.token_ppls)} tokens\n"
    f"    text={final.cum_text[:60]!r}\n"
    f"    ppls={[round(p, 2) for p in final.token_ppls[:8]]}...",
    flush=True,
)

# 3. 优雅停止：第一个更新后停止
t0 = time.time()
got_stop = False
for upd in be.generate_stream(
    "Once upon a time", max_new_tokens=200, do_sample=False
):
    if not got_stop:
        got_stop = True
        be.stop()
    final = upd
assert final.final
print(
    f"[3] graceful stop OK: stopped at {len(final.token_ppls)} tokens "
    f"in {time.time() - t0:.1f}s",
    flush=True,
)

# 4. KV 前缀缓存：同 prompt 二次生成应命中前缀复用（Llama.generate 内部
#    按最长公共前缀复用 KV cache；prompt 需 ≥16 token 才越过复用阈值）
long_prompt = "Once upon a time, there was a little girl named Lily. " * 2
list(be.generate_stream(long_prompt, max_new_tokens=1, do_sample=False))
note_first = be.last_cache_info
list(be.generate_stream(long_prompt, max_new_tokens=1, do_sample=False))
note_second = be.last_cache_info
print(f"[4] cache note: first={note_first!r} → second={note_second!r}", flush=True)
assert "复用" in note_second, note_second

# 5. 聊天模板：Instruct 模型渲染；base 模型应报错并提示 prefix/raw
if has_template:
    p = be.apply_chat_template(
        [{"role": "user", "content": "你好"}], "续写"
    )
    assert "你好" in p and "续写" in p
    print(f"[5] chat template render OK: {p[:60]!r}...", flush=True)
else:
    try:
        be.apply_chat_template([{"role": "user", "content": "你好"}], "")
        raise AssertionError("无模板应报错")
    except ValueError as e:
        assert "prefix/raw" in str(e)
        print(f"[5] no-template error OK: {str(e)[:60]}...", flush=True)

# 6. prefix/raw 模式平文本透传
assert be.build_flat_prompt("ABC") == "ABC"
print("[6] flat prompt passthrough OK")

print("ALL LLAMACPP MODEL TESTS PASSED", flush=True)

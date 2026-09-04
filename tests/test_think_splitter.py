# 思维链分离器 _ThinkSplitter 回归测试（纯逻辑，无需模型/gradio/torch）。
#
# 覆盖两种起始形态：
#   A 开标签在生成流内（DeepSeek-R1 / Qwen3 默认输出）
#   B 开标签已在 prompt 内（Qwen3 系模板 enable_thinking=true 时渲染出未闭合的 <think>）
# 形态 B 若误用 probe 态，思考内容会被整段当作正文、闭标签还会泄漏进正文。
#
# 用法：python tests/test_think_splitter.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import _ThinkSplitter, _prompt_pending_think, _THINK_OPEN, _THINK_CLOSE  # noqa: E402

O = _THINK_OPEN
C = _THINK_CLOSE


def feed_all(splitter, chunks):
    """喂完所有分片并 flush，返回累计 (思维链, 正文)。"""
    r = c = ""
    for ch in chunks:
        a, b = splitter.feed(ch)
        r += a
        c += b
    a, b = splitter.flush()
    return r + a, c + b


def case(name, assume, chunks, want_r, want_c):
    r, c = feed_all(_ThinkSplitter(assume_thinking=assume), chunks)
    assert r == want_r, f"{name}: reasoning {r!r} != {want_r!r}"
    assert c == want_c, f"{name}: content {c!r} != {want_c!r}"
    print(f"{name} OK")


# ----------------------------------------------------------- 形态 A：流内含开标签
case("[1] A 完整 think 块", False,
     [f"{O}我在思考{C}\n\n答案是2。"],
     "我在思考", "\n\n答案是2。")

case("[2] A 标签跨分片边界", False,
     ["<th", "ink", ">我在思考</thi", "nk>\n\n答案是2。"],
     "我在思考", "\n\n答案是2。")

case("[3] A 逐字符喂入", False,
     list(f"{O}思考{C}正文"),
     "思考", "正文")

# ------------------------------------------- 形态 B：开标签已在 prompt 内（回归重点）
case("[4] B 流首即思考正文", True,
     ["我在思", "考...", C, "\n\n答案是2。"],
     "我在思考...", "\n\n答案是2。")

case("[5] B 模板惯用形态（<think> 后带换行）", True,
     ["\n让我想想。", C, "\n\n结论。"],
     "\n让我想想。", "\n\n结论。")

case("[6] B 单分片直达", True,
     [f"思考内容{C}正文内容"],
     "思考内容", "正文内容")

# ------------------------------------------------------------------ 无标签 / 退化
case("[7] 无标签：全正文", False,
     ["直接回答，", "没有思考。"],
     "", "直接回答，没有思考。")

case("[8] 空 think 块（模板关闭思考）", False,
     [f"{O}{C}直接正文"],
     "", "直接正文")

case("[9] B 空思考直接出正文", True,
     [f"{C}直接正文"],
     "", "直接正文")

# --------------------------------------------------- 未闭合：停止生成 / 流被截断
case("[10] A 思考未闭合：尾部归思维链", False,
     [f"{O}想了一半还没"],
     "想了一半还没", "")

case("[11] B 思考未闭合：尾部归思维链", True,
     ["想了一半还没"],
     "想了一半还没", "")

case("[12] probe 态残留不足标签长度", False,
     ["<thi"],
     "", "<thi")

# --------------------------------------------------------- 正文起点坐标（token 对齐用）
print()
s = _ThinkSplitter(assume_thinking=False)
s.feed(f"{O}RR{C}正文")
assert s.content_start() == len(O) + len("RR") + len(C), s.content_start()
print(f"[13] A content_start OK: {s.content_start()}")

s = _ThinkSplitter(assume_thinking=True)
s.feed(f"RR{C}正文")
# 开标签在 prompt 内，不占流内坐标
assert s.content_start() == len("RR") + len(C), s.content_start()
print(f"[14] B content_start OK: {s.content_start()}（开标签不计入流内坐标）")

s = _ThinkSplitter(assume_thinking=False)
s.feed("无标签正文")
assert s.content_start() == 0, s.content_start()
print("[15] 无标签 content_start OK: 0")

# ------------------------------------------------------- prompt 尾部未闭合标签检测
print()
cases = [
    ("<|im_start|>assistant\n" + O + "\n", True, "Qwen3 enable_thinking=true"),
    ("<|im_start|>assistant\n" + O + "\n\n" + C + "\n\n", False, "空 think 块=关闭思考"),
    ("plain prompt without tag", False, "无标签"),
    ("", False, "空 prompt"),
    ("assistant\n" + O, True, "开标签恰在末尾"),
]
for prompt, want, desc in cases:
    got = _prompt_pending_think(prompt)
    assert got == want, f"pending_think({desc}): {got} != {want}"
    print(f"[16] pending_think OK: {str(got):5} <- {desc}")

# _prompt_pending_think 对非字符串输入应安全返回 False
assert _prompt_pending_think(None) is False
assert _prompt_pending_think(123) is False
print("[17] pending_think 非字符串输入 OK")

# 续写场景回归：回灌的思维链可能很长，开标签会落在任何固定大小的尾部
# 窗口之外 → 必须全文扫描，否则模型续写的思考内容会被误判为正文
long_cot = "思考内容" * 300
p_long = "<|im_start|>assistant\n" + O + "\n" + long_cot
assert _prompt_pending_think(p_long), "长思维链：开标签在固定窗口外也必须检出"
assert not _prompt_pending_think(p_long + C + "\n\n正文"), "闭合后不应再判定为思考中"
print(f"[18] pending_think 长思维链（{len(p_long)} 字符）OK")

# splice 的真实用法：模板刚渲染出未闭合开标签，把待续写的思维链放进去。
# 必须是**闭合**的思考区：未闭合回灌会让模型在思考区里直接吐正文且不再补
# 闭标签（Qwen3.5-4B 实测），正文会被整段算进思维链、正文区却是空的。
from backend import _splice_pending_think  # noqa: E402

p_spliced = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", long_cot)
assert p_spliced.count(O) == 1 and p_spliced.count(C) == 1, p_spliced[-60:]
assert p_spliced.endswith(O + "\n" + long_cot + "\n" + C), p_spliced[-40:]
assert not _prompt_pending_think(p_spliced), "闭合回灌后应判定为不在思考区"
# 已闭合的空思考块（开关关闭时的模板形态）→ 换进真实思维链，保留其后空白
spliced2 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n\n" + C + "\n\n", "想了一半")
assert spliced2 == "<|im_start|>assistant\n" + O + "\n想了一半\n" + C + "\n\n", repr(spliced2)
assert not _prompt_pending_think(spliced2), "空思考块填充后仍是闭合的"
print("[19] splice 闭合回灌（长思维链/空思考块）OK")

# 新格式（所见即所得）：cot 自包含思考区标签，直接替换模板思考区——
# 保留闭标签 → 闭合（直接出正文）；删掉闭标签 → 未闭合（续写思考）
p3 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", O + "\n想了一半")
assert p3.endswith(O + "\n想了一半"), p3[-40:]
assert _prompt_pending_think(p3), "未闭合 cot 替换后应处于思考区（续写思考）"
p4 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", O + "\n想了一半\n" + C)
assert p4.endswith(O + "\n想了一半\n" + C), p4[-40:]
assert not _prompt_pending_think(p4), "闭合 cot 替换后不在思考区（直接出正文）"
# 模板未渲染思考区 → 原样接上
p5 = _splice_pending_think("plain prompt", O + "\n想了一半")
assert p5 == "plain prompt\n" + O + "\n想了一半", p5[-40:]
print("[20] splice 新格式（cot 自包含标签，所见即所得）OK")

# KV 缓存复用的关键：重建的思考区必须保持模板 "<think>\n" 结构。
# 原实现把 cot 内容（<think> 后直接跟正文，无换行）原样替换 → token
# 序列在 <think> 后即与上次生成的 "<think>\n…" 前缀分叉（实测只匹配
# 模板部分 22/322）→ KV 复用失效 → 回灌整段思维链全量 prefill，随
# 长度线性变慢。重建后前缀恢复一致，复用生效（322/322 完全命中）。
p6 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", O + "想了一半")
assert p6 == "<|im_start|>assistant\n" + O + "\n想了一半", repr(p6)
assert _prompt_pending_think(p6), "未闭合重建后应在思考区"
p7 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", O + "想了一半\n" + C)
assert p7 == "<|im_start|>assistant\n" + O + "\n想了一半\n" + C, repr(p7)
assert not _prompt_pending_think(p7), "闭合重建后不在思考区"
# 多行内容：strip 只去首尾换行，中间换行保留
p8 = _splice_pending_think("<|im_start|>assistant\n" + O + "\n", O + "\n第一行\n第二行\n" + C)
assert p8 == "<|im_start|>assistant\n" + O + "\n第一行\n第二行\n" + C, repr(p8)
print("[21] splice 新格式保持 <think>\\n 结构（KV 缓存复用）OK")

print("\nALL THINK SPLITTER TESTS PASSED")

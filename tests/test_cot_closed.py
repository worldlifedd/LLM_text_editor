# -*- coding: utf-8 -*-
"""验证 normalize_cot / build_prompt 新旧格式。"""
import sys
sys.path.insert(0, r"e:\myGithub\LLM_text_editor")
from core import build_prompt, normalize_cot

O = "<" + "thi" + "nk>"
C = "</" + "thi" + "nk>"

class FakeBackend:
    def build_chat_prompt(self, msgs, active, enable_thinking=None, cot_prefix=""):
        return "CHAT[" + "|".join(m["role"] + ":" + m["content"] for m in msgs) + "]" + \
               f"|active={active}|cot={cot_prefix!r}"
    def build_flat_prompt(self, flat):
        return "FLAT[" + flat + "]"

# 新格式：content=正文 + closed
b_new_closed = {"type": "cot", "content": "想完了", "closed": True}
b_new_open = {"type": "cot", "content": "想到一半", "closed": False}
# 旧格式：标签在内容里
b_old_closed = {"type": "cot", "content": O + "\n旧想完" + C}
b_old_open = {"type": "cot", "content": O + "\n旧想到一半"}
# 纯文本历史
b_plain = {"type": "cot", "content": "纯文本思考"}

for name, blk, exp_cot in [
    ("new-closed", b_new_closed, O + "\n想完了\n" + C),
    ("new-open", b_new_open, O + "\n想到一半"),
    ("old-closed", b_old_closed, O + "\n旧想完\n" + C),
    ("old-open", b_old_open, O + "\n旧想到一半"),
    ("plain", b_plain, O + "\n纯文本思考\n" + C),
]:
    got = normalize_cot(blk)
    print(f"[{name}] normalize -> body={got[0]!r} closed={got[1]}")
    assert got == ({"new-closed": ("想完了", True), "new-open": ("想到一半", False),
                    "old-closed": ("旧想完", True), "old-open": ("旧想到一半", False),
                    "plain": ("纯文本思考", True)}[name]), name

# build_prompt chat 模式：cot 回灌
for name, blk, expect_closed in [
    ("new-closed", b_new_closed, True), ("new-open", b_new_open, False),
    ("old-closed", b_old_closed, True), ("old-open", b_old_open, False),
]:
    blocks = [{"type": "prompt", "content": "问"}, blk, {"type": "generate", "content": ""}]
    p = build_prompt(blocks, "", skill_ctx="", mode="chat", backend=FakeBackend())
    assert (C in p) == expect_closed, (name, p)
    print(f"[chat {name}] closed_in_cot={C in p} OK: {p[p.find('|cot='):][:80]}")

# flat 模式：head 不再双重开标签
blocks = [{"type": "prompt", "content": "问"}, b_new_open, {"type": "generate", "content": ""}]
p = build_prompt(blocks, "", skill_ctx="", mode="raw", backend=FakeBackend())
assert p.count(O) == 1, p  # 旧实现会是 2（OPEN + 内容自带 OPEN）
print(f"[flat new-open] single-open OK: {p[-70:]}")

# active 非空：cot 仍回灌（当前回复的思考区，续写上下文与首次生成一致）
blocks = [{"type": "prompt", "content": "问"}, b_new_open, {"type": "generate", "content": "已有正文"}]
p = build_prompt(blocks, "已有正文", skill_ctx="", mode="chat", backend=FakeBackend())
assert "cot=" + repr(O + "\n想到一半") in p, p
print("[chat active非空] cot回灌（未闭合原样） OK")
blocks = [{"type": "prompt", "content": "问"}, b_new_closed, {"type": "generate", "content": "已有正文"}]
p = build_prompt(blocks, "已有正文", skill_ctx="", mode="chat", backend=FakeBackend())
assert "cot=" + repr(O + "\n想完了\n" + C) in p, p
print("[chat active非空] cot回灌（闭合包装） OK")
print("ALL PASS")

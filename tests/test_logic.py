# -*- coding: utf-8 -*-
"""纯逻辑冒烟测试：文档序列化、技能解析、困惑度可视化辅助函数。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
import core
from skills import scan_skills, skills_to_context

# 1. 序列化/解析 roundtrip
blocks = [
    {"type": "prompt", "content": "写一段散文。"},
    {"type": "generate", "content": "秋风起了。"},
    {"type": "prompt", "content": "继续写雪景。"},
]
active = "冬雪落了下来。"
text = app.serialize_doc(blocks, active)
assert "<!-- prompt" in text and "<!-- generate -->" in text, text
assert "秋风起了。" in text and "冬雪落了下来。" in text  # 生成块为可见正文
b2, a2 = app.parse_doc(text)
assert len(b2) == 3 and b2[0]["content"] == "写一段散文。" and a2 == active, (b2, a2)
# 无活动文本时：最后一块是 prompt，全部留在历史，活动单元为空
b3, a3 = app.parse_doc(app.serialize_doc(blocks, ""))
assert len(b3) == 3 and a3 == "", (b3, a3)
# 仅有活动文本：全部进活动单元
b4, a4 = app.parse_doc(app.serialize_doc([], "只有活动文本。"))
assert b4 == [] and a4 == "只有活动文本。", (b4, a4)
# 空生成块（定稿后的空活动单元）以 <!-- generate --> 标记落盘保留
t5 = app.serialize_doc([{"type": "prompt", "content": "p"}, {"type": "generate", "content": ""}], "")
assert "<!-- generate -->" in t5, t5
b5, a5 = app.parse_doc(t5)
assert b5 == [{"type": "prompt", "content": "p"}] and a5 == "", (b5, a5)
# 手写纯 Markdown：注释为提示词，可见文本为生成内容
hand = "开头裸文本\n\n<!-- prompt\n手写指令\n-->\n\n手写正文\n"
hb, ha = app.parse_doc(hand)
assert hb == [{"type": "generate", "content": "开头裸文本"},
              {"type": "prompt", "content": "手写指令"}], hb
assert ha == "手写正文", ha
# 相邻生成块以 <!-- generate --> 标记分界（最后一块进活动单元）
adj = app.parse_doc("<!-- generate -->\n甲\n\n<!-- generate -->\n乙\n\n<!-- generate -->\n丙\n")
assert adj[0] == [{"type": "generate", "content": "甲"},
                  {"type": "generate", "content": "乙"}], adj
assert adj[1] == "丙", adj[1]
# 系统块（固化技能）：<!-- system ... --> 注释，roundtrip 保留
sys_blocks = [
    {"type": "system", "content": "# 技能指令: 中文散文写作\n写优美的散文。"},
    {"type": "prompt", "content": "写一段散文。"},
]
ts = app.serialize_doc(sys_blocks, "秋风起了。")
assert "<!-- system" in ts, ts
bs, as_ = app.parse_doc(ts)
assert bs == sys_blocks and as_ == "秋风起了。", (bs, as_)
# 上下文组装：system 块与本地技能合并为系统上下文，其余块不变
sys_ctx, rest = core.extract_system(
    [{"type": "system", "content": "SYS_IN_DOC"},
     {"type": "prompt", "content": "A"},
     {"type": "system", "content": ""},
     {"type": "generate", "content": "B"}],
    "LOCAL_SKILL",
)
assert sys_ctx == "SYS_IN_DOC\n\nLOCAL_SKILL", sys_ctx
assert [b["type"] for b in rest] == ["prompt", "generate"], rest
assert core.extract_system([{"type": "prompt", "content": "A"}], "") == ("", [{"type": "prompt", "content": "A"}])
# 思维链块：<!-- cot ... --> 注释，roundtrip 保留且不进上下文
cot_blocks = [
    {"type": "cot", "content": "先构思结构，再落笔。"},
    {"type": "prompt", "content": "写一段散文。"},
]
tc = app.serialize_doc(cot_blocks, "秋风起了。")
assert "<!-- cot" in tc, tc
bc, ac = app.parse_doc(tc)
assert bc == cot_blocks and ac == "秋风起了。", (bc, ac)


class _FlatBackend:
    kind = "test"

    def build_chat_prompt(self, msgs, active, enable_thinking=None, cot_prefix=""):
        return ("chat", msgs, active, cot_prefix)

    def build_flat_prompt(self, flat):
        return ("flat", flat)


# cot 不进上下文（推理模型会自行重新思考）
_O, _C = "<" + "thi" + "nk>", "</" + "thi" + "nk>"
p = core.build_prompt(
    [{"type": "cot", "content": "SECRET_COT"},
     {"type": "prompt", "content": "写散文"}],
    "", "", "raw", _FlatBackend(),
)
assert "SECRET_COT" not in str(p), p
p2 = core.build_prompt(
    [{"type": "prompt", "content": "写散文"},
     {"type": "cot", "content": "SECRET_COT"},
     {"type": "generate", "content": "秋。"}],
    "续写", "", "prefix", _FlatBackend(),
)
# 定稿后续写：思考属本轮（generate 不断链）→ 仍作为带标签思考区回灌，
# 但绝不作为普通块内容进上下文
assert _O + "\nSECRET_COT\n" + _C in p2[1], p2
assert "正文】\nSECRET_COT" not in p2[1], p2
# 中断续写：正文尚未开始时，紧邻活动单元的 cot 应作为续写头回灌，
# 否则再次生成会让模型从零重想一遍（表现为重复输出思考开头）
p3 = core.build_prompt(
    [{"type": "prompt", "content": "写散文"},
     {"type": "cot", "content": "已想了一半"}],
    "", "", "chat", _FlatBackend(),
)
# 新契约：cot_prefix 恒为带标签的思考区（closed 属性/旧格式标签推导闭合态，
# 纯文本历史块按已闭合回灌），backend _splice_pending_think 负责替换模板思考区
assert p3[3] == _O + "\n已想了一半\n" + _C, p3  # cot_prefix 回灌（闭合包装）
assert "已想了一半" not in str(p3[1]), p3   # 但不作为普通块进 messages
# 正文已开始 → 仍回灌：活动单元的思考区与正文同属当前 assistant 回复，
# 续写上下文必须与首次生成一致（否则困惑度着色漂移、KV 缓存整体失效）
p4 = core.build_prompt(
    [{"type": "prompt", "content": "写散文"},
     {"type": "cot", "content": "已想完"}],
    "已有正文", "", "chat", _FlatBackend(),
)
assert p4[3] == _O + "\n已想完\n" + _C, p4
# 未闭合 + 正文已开始：按未闭合回灌（所见即所得，模型从断点继续思考）
p4o = core.build_prompt(
    [{"type": "prompt", "content": "写散文"},
     {"type": "cot", "content": "想了一半", "closed": False}],
    "已有正文", "", "chat", _FlatBackend(),
)
assert p4o[3] == _O + "\n想了一半", p4o
# raw 模式：思考区在活动正文之前（顺序）
p6 = core.build_prompt(
    [{"type": "prompt", "content": "写散文"},
     {"type": "cot", "content": "思路"}],
    "已有正文", "", "raw", _FlatBackend(),
)
flat = p6[1]
assert flat.index("思路") < flat.index("已有正文") and "写散文" in flat, flat
# cot 与活动单元之间隔着 prompt → 不是本轮的思考，不回灌
p5 = core.build_prompt(
    [{"type": "cot", "content": "旧思考"},
     {"type": "prompt", "content": "换个题目"}],
    "", "", "chat", _FlatBackend(),
)
assert p5[3] == "", p5
print("[1] serialize/parse roundtrip OK")

# 2. 技能扫描 + frontmatter + 上下文拼接
skills = scan_skills()
assert len(skills) >= 1, "应有示例技能"
s = skills[0]
assert s.name == "中文散文写作" and "散文" in s.instructions and s.description, s
ctx = skills_to_context(skills)
assert "技能指令" in ctx and "中文散文写作" in ctx
assert skills_to_context([]) == ""
print("[2] skills scan/parse/context OK ->", [x.name for x in skills])

# 3. 伪彩色与热力图（仅着色辅助 + 灰显 span，不再依赖旧 heatmap_html 签名）
c1, c200 = app.ppl_color(1), app.ppl_color(200)
assert c1.startswith("rgba(134") and c200.startswith("rgba(255"), (c1, c200)
# 使用 core 函数直接验证：token_texts + token_ppls 的着色拼接
texts, ppls = ["秋风", " 起了", "\n落叶"], [1.2, 30.0, 150.0]
spans = [
    f'<span style="background:{app.ppl_color(p)}">{t}</span>'
    for t, p in zip(texts, ppls)
]
hm = "".join(spans)
assert "background:rgba" in hm and "落叶" in hm
# doc_heatmap_html 空数据应有占位
empty_hm = app.doc_heatmap_html([], "")
assert empty_hm.startswith("<i"), "空数据应有占位"
print("[3] heatmap/color OK")

# 4. 曲线数据
df = app.plot_data([1.0, 5.0, 20.0])
assert len(df) == 6 and {"index", "ppl", "type"} <= set(df.columns)
assert len(app.plot_data([])) == 0
assert math.isclose(app.avg_ppl_of([4.0, 9.0]), 6.0, rel_tol=1e-9)
assert app.avg_ppl_of([]) is None
print("[4] plot/avg OK")

# 5. 上下文组装（使用 core.build_flat_context 替代旧 build_context）
ctx = core.build_flat_context(
    [{"type": "prompt", "content": "A"}, {"type": "generate", "content": "B"}], "C", "SKL", "raw"
)
assert ctx == "SKL\n\nA\n\nB\n\nC", ctx
print("[5] build_flat_context OK")

# 6. 着色对账 reconcile_active_ppl：返回值必须覆盖整个 base（不变量）
# 完全覆盖 → 原样返回
assert core.reconcile_active_ppl(["秋", "风"], [1.2, 3.4], "秋风") == (["秋", "风"], [1.2, 3.4])
# 覆盖是 base 严格前缀（末尾追加文本）→ 补灰段（此前不补 → 与后续段
# 拼接出现字符空洞，生成时整块着色消失）
assert core.reconcile_active_ppl(["秋"], [1.2], "秋风起") == (["秋", "风起"], [1.2, None])
# 中点编辑 → 公共前缀内完整 token 保留，其后合并为灰段
assert core.reconcile_active_ppl(["秋风", "起了"], [1.2, 3.4], "秋风来了") == (["秋风", "来了"], [1.2, None])
# 空段 → 空
assert core.reconcile_active_ppl([], [], "秋风") == ([], [])
print("[6] reconcile_active_ppl coverage OK")

print("ALL LOGIC TESTS PASSED")

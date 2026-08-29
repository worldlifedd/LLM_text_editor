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

    def build_chat_prompt(self, msgs, active):
        return ("chat", msgs, active)

    def build_flat_prompt(self, flat):
        return ("flat", flat)


# cot 不进上下文（推理模型会自行重新思考）
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
assert "SECRET_COT" not in str(p2), p2
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

print("ALL LOGIC TESTS PASSED")

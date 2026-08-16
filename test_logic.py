# -*- coding: utf-8 -*-
"""纯逻辑冒烟测试：文档序列化、技能解析、困惑度可视化辅助函数。"""
import math

import app
from skills import scan_skills, skills_to_context

# 1. 序列化/解析 roundtrip
blocks = [
    {"type": "prompt", "content": "写一段散文。"},
    {"type": "generate", "content": "秋风起了。"},
    {"type": "prompt", "content": "继续写雪景。"},
]
active = "冬雪落了下来。"
text = app.serialize_doc(blocks, active)
assert "<prompt>" in text and "<generate>" in text, text
b2, a2 = app.parse_doc(text)
assert len(b2) == 3 and b2[0]["content"] == "写一段散文。" and a2 == active, (b2, a2)
# 无活动文本时：最后一块是 prompt，全部留在历史，活动单元为空
b3, a3 = app.parse_doc(app.serialize_doc(blocks, ""))
assert len(b3) == 3 and a3 == "", (b3, a3)
# 仅有活动文本：全部进活动单元
b4, a4 = app.parse_doc(app.serialize_doc([], "只有活动文本。"))
assert b4 == [] and a4 == "只有活动文本。", (b4, a4)
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

# 3. 伪彩色与热力图
c1, c200 = app.ppl_color(1), app.ppl_color(200)
assert c1.startswith("rgba(134") and c200.startswith("rgba(255"), (c1, c200)
hm = app.heatmap_html(["秋风", " 起了", "\n落叶"], [1.2, 30.0, 150.0])
assert "background:rgba" in hm and "落叶" in hm and "&nbsp;" in hm and "<br" not in hm
assert app.heatmap_html([], []) .startswith("<i"), "空数据应有占位"
print("[3] heatmap/color OK")

# 4. 曲线数据
df = app.plot_data([1.0, 5.0, 20.0])
assert len(df) == 6 and {"index", "ppl", "type"} <= set(df.columns)
assert len(app.plot_data([])) == 0
assert abs(app.avg_ppl_of([4.0, 9.0]) - 6.0) < 1e-6 or True  # sqrt(36)=6
assert math.isclose(app.avg_ppl_of([4.0, 9.0]), 6.0, rel_tol=1e-9)
assert app.avg_ppl_of([]) is None
print("[4] plot/avg OK")

# 5. 上下文组装
ctx = app.build_context(
    [{"type": "prompt", "content": "A"}, {"type": "generate", "content": "B"}], "C", "SKL"
)
assert ctx == "SKL\n\nA\n\nB\n\nC", ctx
print("[5] build_context OK")

print("ALL LOGIC TESTS PASSED")

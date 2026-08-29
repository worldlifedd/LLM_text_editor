# -*- coding: utf-8 -*-
"""困惑度跨块累积验证：两轮生成 + 定稿冻结 + 累积展示 + 编辑失效。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GEN = dict(max_new_tokens=24, do_sample=False, temperature=0.8,
           top_k=50, top_p=0.95, repetition_penalty=1.1)

print("[0] 加载模型 ...")
for out in app.on_load_model("local", MODEL, "", "", ""):
    st = out[app.model_status]
assert st.startswith("✅"), st
print(f"    {st}")

blocks = [{"type": "prompt", "content": "写一段关于秋天的散文开头，50字左右。"}]


def colored_spans(hm):
    """正文着色 span 数（灰显 160,160,160 与图例 3 个伪彩色块除外）。"""
    import re as _re
    return len(_re.findall(r'<span style="background:rgba\((?!160,160,160)', hm)) - 3


def gen_once(blocks, active, tag):
    final = None
    for out in app.on_generate(blocks, active, [], [], *GEN.values(), "chat"):
        final = out
    print(f"    [{tag}] {final[app.status_tb]}")
    return final


print("[1] 第一轮生成 ...")
f1 = gen_once(blocks, "", "轮1")
text1 = f1[app.active_cell_tb]
assert len(text1) > 5
n1 = len(app._ACTIVE_PPL["token_ppls"])
assert n1 > 0, "活动块着色数据为空"
hm1 = f1[app.heatmap_md]
c1 = colored_spans(hm1)
assert c1 == n1, f"热力图着色数 {c1} != token 数 {n1}"

print("[2] 定稿冻结（着色随块保留）...")
fr = app.on_add_generate(blocks, text1)
blocks2 = fr[app.blocks_state]
assert len(blocks2) == 2 and blocks2[1].get("ppl"), "冻结块未携带 ppl"
assert not app._ACTIVE_PPL["token_texts"], "活动着色未重置"
hm_frozen = fr[app.heatmap_md]
assert colored_spans(hm_frozen) == n1, "定稿后着色丢失"

print("[3] 第二轮生成（新活动块，历史块着色保留）...")
blocks3 = blocks2 + [{"type": "prompt", "content": "继续写冬天的部分，50字。"}]
f2 = gen_once(blocks3, "", "轮2")
text2 = f2[app.active_cell_tb]
assert len(text2) > 5
n2 = len(app._ACTIVE_PPL["token_ppls"])
hm2 = f2[app.heatmap_md]
c2 = colored_spans(hm2)
assert c2 == n1 + n2, f"跨块着色数 {c2} != {n1}+{n2}"
assert hm2.count("▍") >= 1, "缺少块边界分隔符"

print("[4] 续写第三轮（活动块文本 = 轮2文本，着色前缀对账保留）...")
f3 = gen_once(blocks3, text2, "轮3续写")
n3 = len(app._ACTIVE_PPL["token_ppls"])
assert n3 > n2, "续写未累积着色"

print("[5] 中段编辑一字 → 仅编辑点之后失效，前缀着色保留 ...")
text3 = f3[app.active_cell_tb]
i_mid = len(text3) // 2
edited_mid = text3[:i_mid] + "改" + text3[i_mid + 1:]
f4 = gen_once(blocks3, edited_mid, "中段编辑")
ppls4 = app._ACTIVE_PPL["token_ppls"]
assert None in ppls4, "编辑区域应出现无数据段"
first_none = ppls4.index(None)
assert first_none > 5, f"编辑点之前的着色应保留（首个 None 在 {first_none}）"
assert all(p is not None for p in ppls4[:first_none]), "前缀着色损坏"
assert all(p is not None for p in ppls4[first_none + 1:]), "新生成部分未着色"
# 不变量：token_texts 拼接 == 活动块完整文本
assert "".join(app._ACTIVE_PPL["token_texts"]) == f4[app.active_cell_tb], "覆盖拼接与文本不对齐"
hm4 = f4[app.heatmap_md]
assert "灰=手动编辑/无数据" in hm4
assert colored_spans(hm4) > 0, "编辑后前缀着色全部丢失"

print("[6] 开头编辑 → 无公共前缀，旧着色全部失效 ...")
edited_head = "完全不同的开头。" + edited_mid[10:]
f5 = gen_once(blocks3, edited_head, "开头编辑")
ppls5 = app._ACTIVE_PPL["token_ppls"]
assert ppls5[0] is None, "开头编辑后首段应为无数据段"
assert all(p is not None for p in ppls5[1:]), "新生成部分未着色"
assert "".join(app._ACTIVE_PPL["token_texts"]) == f5[app.active_cell_tb]

print("[7] 无 ppl 块灰显 + prompt 块不进热力图 ...")
manual_blocks = [
    {"type": "prompt", "content": "提示词不该出现"},
    {"type": "generate", "content": "手动写的生成块"},
]
hm6 = app.doc_heatmap_html(manual_blocks, "手动活动文本")
assert "手动写的生成块" in hm6 and "rgba(160,160,160" in hm6
assert "提示词不该出现" not in hm6

print("[8] 序列化兼容（ppl 键不写入文件）...")
doc = app.serialize_doc(blocks3, text2)
assert "token_ppls" not in doc and "<!-- generate -->" in doc

print("ALL PPL ACCUMULATION TESTS PASSED ✔")

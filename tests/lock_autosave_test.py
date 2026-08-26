# -*- coding: utf-8 -*-
"""块锁定 + 定时自动保存逻辑验证（无需模型/服务器）。"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

print("[1] 锁定/解锁翻转 ...")
blocks = [{"type": "prompt", "content": "a"},
          {"type": "generate", "content": "b"}]
b = app.toggle_block_lock(blocks, 0)
assert b[0]["locked"] is True and b[1].get("locked") is None
b2 = app.toggle_block_lock(b, 0)
assert b2[0]["locked"] is False
r = app.toggle_block_lock(blocks, 99)  # 越界 → gr.skip
assert r is not None
print("    ✔")

print("[2] on_add_generate 自动锁定前序块 ...")
blocks2 = [{"type": "prompt", "content": "p1"},
           {"type": "generate", "content": "g1"},
           {"type": "prompt", "content": "p2"}]
res = app.on_add_generate(blocks2, "生成的文本")
nb = res[app.blocks_state]
assert len(nb) == 4
assert all(nb[i].get("locked") for i in range(3)), "前序块未锁定"
assert not nb[3].get("locked"), "新定稿块不应锁定"
print("    ✔ 3 个前序块已锁定，新块未锁定")

print("[3] 空文档定稿追加空生成块，不报错 ...")
res2 = app.on_add_generate([], "")
assert len(res2[app.blocks_state]) == 1
assert res2[app.blocks_state][0]["type"] == "generate"
assert res2[app.blocks_state][0]["content"] == ""

print("[4] 定时自动保存：落盘 + 内容去重 + 开关 + 轮转 ...")
with tempfile.TemporaryDirectory() as td:
    old_dir, app.SAVE_DIR = app.SAVE_DIR, td
    app._LAST_AUTOSAVE_TEXT = None
    try:
        r1 = app.on_autosave([{"type": "prompt", "content": "x"}], "y", True)
        assert len(glob.glob(os.path.join(td, "autosave_*.md"))) == 1
        # 内容无变化 → 跳过
        r2 = app.on_autosave([{"type": "prompt", "content": "x"}], "y", True)
        assert len(glob.glob(os.path.join(td, "autosave_*.md"))) == 1
        assert "跳过" in r2[app.autosave_tb]
        # 内容变化 → 新文件
        r3 = app.on_autosave([{"type": "prompt", "content": "x2"}], "y", True)
        n3 = len(glob.glob(os.path.join(td, "autosave_*.md")))
        print(f"    debug: r3 后文件数 {n3}, r3={r3[app.autosave_tb]}")
        assert n3 == 2
        assert "已自动保存" in r3[app.autosave_tb]
        # 开关关闭 → 不保存
        r4 = app.on_autosave([{"type": "prompt", "content": "z"}], "y", False)
        assert r4[app.autosave_tb] == ""
        # 空文档 → 不保存
        r5 = app.on_autosave([], "", True)
        assert r5[app.autosave_tb] == ""
        # 轮转：最多保留 _AUTOSAVE_KEEP 份
        app._LAST_AUTOSAVE_TEXT = None
        for i in range(40):
            app._LAST_AUTOSAVE_TEXT = None
            app.on_autosave([{"type": "prompt", "content": f"x{i}"}], "y", True)
        n = len(glob.glob(os.path.join(td, "autosave_*.md")))
        assert n == app._AUTOSAVE_KEEP, f"轮转失败：{n} != {app._AUTOSAVE_KEEP}"
        # 文件内容合法（XML 标记）
        with open(glob.glob(os.path.join(td, "autosave_*.md"))[0], encoding="utf-8") as f:
            assert "<prompt>" in f.read()
    finally:
        app.SAVE_DIR = old_dir
print("    ✔ 落盘/去重/开关/空文档/轮转均通过")

print("[5] 序列化不包含 locked 键 ...")
doc = app.serialize_doc([{"type": "prompt", "content": "p1", "locked": True}], "")
assert "locked" not in doc and "<prompt>" in doc

print("ALL LOCK + AUTOSAVE TESTS PASSED ✔")

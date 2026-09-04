# -*- coding: utf-8 -*-
"""API 级验证：md 导入（含标签 cot）→ 归一化 round-trip。"""
import json
import sys
import urllib.request

sys.path.insert(0, r"e:\myGithub\LLM_text_editor")

O = "<" + "thi" + "nk>"
C = "</" + "thi" + "nk>"
MD = f"""<!-- prompt
问
-->

<!-- cot
{O}
思考正文
{C}
-->

<!-- cot
{O}
未结束思考
-->
"""


def post(path, payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8908" + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.load(urllib.request.urlopen(req))


doc = post("/api/docs/import_md", {"text": MD, "title": "import-tags"})
blocks = doc["blocks"]
print("cot1:", repr(blocks[1]["content"]), "closed =", blocks[1].get("closed"))
print("cot2:", repr(blocks[2]["content"]), "closed =", blocks[2].get("closed"))
assert blocks[1]["content"] == "思考正文" and blocks[1]["closed"] is True
assert blocks[2]["content"] == "未结束思考" and blocks[2]["closed"] is False
print("IMPORT ROUND-TRIP PASS")

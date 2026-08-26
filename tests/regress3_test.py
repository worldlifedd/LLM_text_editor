# -*- coding: utf-8 -*-
"""服务器回归：加载 + 连续两次生成（验证缓存路径在 UI 链路正常）。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = "http://127.0.0.1:7860"
S = requests.Session()


def call(api, data, timeout=300):
    r = S.post(f"{BASE}/gradio_api/call/{api}", json={"data": data}, timeout=30)
    r.raise_for_status()
    eid = r.json()["event_id"]
    with S.get(f"{BASE}/gradio_api/call/{api}/{eid}", stream=True, timeout=timeout) as resp:
        events, buf = [], None
        for line in resp.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("event:"):
                buf = line[6:].strip()
            elif line.startswith("data:") and buf:
                payload = line[5:].strip()
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
                events.append((buf, payload))
                if buf in ("complete", "error"):
                    break
        return events


def unwrap(events):
    for ev, payload in reversed(events):
        if ev == "error":
            raise RuntimeError(f"API error: {payload}")
        if ev == "complete" and isinstance(payload, list):
            return payload
    raise RuntimeError("no complete event")


for _ in range(10):
    try:
        requests.get(BASE, timeout=3)
        break
    except Exception:
        time.sleep(3)

blocks = [{"type": "prompt", "content": "写一段关于秋天的散文，80字左右。"}]
print("[1] 加载模型 ...")
st = unwrap(call("on_load_model", ["local", "Qwen/Qwen2.5-0.5B-Instruct", "", "", ""]))[0]
assert st.startswith("✅"), st
print(f"    {st}")

print("[2] 第一次生成（缓存未命中）...")
final = unwrap(call("on_generate", [blocks, "", [], [], 32, True, 0.8, 50, 0.95, 1.1, "chat"]))
text1, status1 = final[0], final[-1]
assert len(text1) > 5, text1
print(f"    {status1}")

print("[3] 第二次生成（同文档续写 → 缓存命中）...")
# 注：REST 调用时 active_cell_tb（双向组件）传参被服务器会话状态覆盖
# （已知 Gradio 限制，浏览器 UI 不受影响），故此处只验证缓存命中与生成成功
final = unwrap(call("on_generate", [blocks, text1, [], [], 32, True, 0.8, 50, 0.95, 1.1, "chat"]))
text2, status2 = final[0], final[-1]
assert len(text2) > 5, "第二次生成失败"
assert "KV缓存复用" in status2, f"缓存未命中: {status2}"
print(f"    {status2}")

print("SERVER REGRESSION PASSED ✔")

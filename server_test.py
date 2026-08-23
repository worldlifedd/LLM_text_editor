# -*- coding: utf-8 -*-
"""无头服务回归：状态/加载 + 连续两次生成（验证 KV 缓存路径）+ 停止。

用法：先启动 `python server.py`，再运行本脚本。
"""
import json
import time

import requests

BASE = "http://127.0.0.1:8907"
S = requests.Session()


def post(api, data, timeout=300):
    r = S.post(f"{BASE}{api}", json=data, timeout=timeout)
    return r


def sse_events(resp):
    """把 SSE 响应解析为 [(event, data_dict)]。"""
    events, buf = [], None
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event:"):
            buf = line[6:].strip()
        elif line.startswith("data:"):
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                payload = line[5:].strip()
            if buf:
                events.append((buf, payload))
                if buf in ("error",) or (buf == "update" and payload.get("final")):
                    break
    return events


def sse_events_generator(resp):
    """逐步 yield (event, data_dict)，调用方可边读边停止。"""
    buf = None
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event:"):
            buf = line[6:].strip()
        elif line.startswith("data:"):
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                payload = line[5:].strip()
            if buf:
                yield buf, payload
                if buf in ("error",) or (buf == "update" and payload.get("final")):
                    return


def wait_server(timeout=60):
    for _ in range(timeout * 2):
        try:
            r = requests.get(f"{BASE}/api/status", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server 未就绪")


def wait_loaded(timeout=600):
    """轮询 /api/status 直至 loaded 且 loading 结束；返回最终 status。"""
    st = None
    for _ in range(timeout * 2):
        st = requests.get(f"{BASE}/api/status", timeout=3).json()
        if st["loading"]:
            time.sleep(0.5)
            continue
        if st["loaded"]:
            return st
        # 加载失败：message 以 ❌ 开头
        if st["message"].startswith("❌"):
            raise RuntimeError(st["message"])
        time.sleep(0.5)
    raise RuntimeError(f"加载超时，最后状态：{st}")


wait_server()
print("[0] 未加载时 generate 应返回错误")
r = post("/api/generate", {"blocks": [{"type": "generate", "content": ""}], "params": {"max_new_tokens": 8}})
assert r.status_code == 400, r.text
print(f"    {r.json()}")

print("[1] 加载模型 ...")
r = post("/api/load", {"mode": "local", "model_path": "Qwen/Qwen2.5-0.5B-Instruct"})
assert r.status_code == 200, r.text
st = wait_loaded()
print(f"    {st['message']}")

print("[2] 第一次生成（缓存未命中）...")
blocks = [{"type": "prompt", "content": "写一段关于秋天的散文，80字左右。"},
          {"type": "generate", "content": ""}]
r = post("/api/generate", {"blocks": blocks, "params": {"max_new_tokens": 32}})
assert r.status_code == 200, r.text
events = sse_events(r)
text1 = None
cache1 = ""
for ev, payload in events:
    if ev == "update":
        text1 = payload["cum_text"]
        cache1 = payload.get("cache_info") or ""
    elif ev == "error":
        raise RuntimeError(payload)
assert text1 and len(text1) > 5, text1
assert "KV缓存" in cache1 and "复用" not in cache1, cache1  # 首次未命中
print(f"    {cache1}｜长度 {len(text1)}")

print("[3] 第二次生成（同文档续写 → 缓存命中）...")
blocks = [{"type": "prompt", "content": "写一段关于秋天的散文，80字左右。"},
          {"type": "generate", "content": text1}]
r = post("/api/generate", {"blocks": blocks, "params": {"max_new_tokens": 32}})
assert r.status_code == 200, r.text
events = sse_events(r)
text2, cache2 = None, ""
for ev, payload in events:
    if ev == "update":
        text2 = payload["cum_text"]
        cache2 = payload.get("cache_info") or ""
    elif ev == "error":
        raise RuntimeError(payload)
assert text2 and len(text2) > 5, text2
assert "复用" in cache2, f"缓存未命中: {cache2}"
print(f"    {cache2}｜长度 {len(text2)}")

print("[4] 生成中停止 ...")
blocks = [{"type": "prompt", "content": "写一段关于秋天的散文，80字左右。"},
          {"type": "generate", "content": ""}]
r = post("/api/generate", {"blocks": blocks, "params": {"max_new_tokens": 1024}})
resp = r
# 读几条 update 后立刻停止
seen = 0
for ev, payload in sse_events_generator(resp):
    if ev == "update" and payload["cum_text"]:
        seen += 1
        if seen >= 3:
            S.post(f"{BASE}/api/stop", timeout=10)
    elif ev == "error":
        break
    if payload.get("final"):
        break
print(f"    停止前收到 {seen} 条 update，流已收尾")

print("SERVER REGRESSION PASSED ✔")

"""复现侧边栏停止生成无效：mock API 慢速流 + server + stop 实测。

- mock OpenAI 兼容服务器：每 0.5s 发一个 chunk，模拟持续生成
- 起 server.py（API 模式），POST /api/generate 后 POST /api/stop
- 轮询 /api/monitor 判断 generating 是否及时变 false
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

MOCK_PORT = 9100
SERVER_PORT = 9101
HALT = threading.Event()

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        stream = body.get("stream")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not stream:
            payload = {
                "id": "mock", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"total_tokens": 1},
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            return
        # 流式：无限产出（每 0.5s 一个 chunk），直到收到连接关闭
        try:
            n = 0
            while not HALT.is_set() and n < 400:
                chunk = {
                    "id": "mock", "object": "chat.completion.chunk", "model": body.get("model"),
                    "choices": [{"index": 0, "delta": {"content": f"tok{n}"},
                                 "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                n += 1
                time.sleep(0.5)
            if n < 400:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开（stop 后 resp.close()）→ 正常结束

    def log_message(self, *a):
        pass


def json_req(url, body=None, method="POST"):
    req = urllib.request.Request(
        url, data=(json.dumps(body).encode() if body is not None else None),
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    mock = HTTPServer(("127.0.0.1", MOCK_PORT), MockHandler)
    threading.Thread(target=mock.serve_forever, daemon=True).start()

    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", str(SERVER_PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base = f"http://127.0.0.1:{SERVER_PORT}"
        # 等服务就绪
        for _ in range(50):
            try:
                json_req(f"{base}/api/status", method="GET")
                break
            except Exception:
                time.sleep(0.2)

        json_req(f"{base}/api/load", {
            "mode": "api", "base_url": f"http://127.0.0.1:{MOCK_PORT}/v1", "model": "mock",
        })
        time.sleep(1.5)

        req = {
            "blocks": [
                {"type": "prompt", "content": "测试停止"},
                {"type": "generate", "content": ""},
            ],
            "active_text": "", "skills": [],
            "params": {"max_new_tokens": 200, "do_sample": False, "temperature": 0.8,
                       "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1,
                       "enable_thinking": None},
            "context_mode": "chat",
        }

        # 发起生成（不读流，直接发请求会阻塞到流结束——放到线程里）
        gen_thread = threading.Thread(
            target=lambda: urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/generate", data=json.dumps(req).encode(),
                    headers={"Content-Type": "application/json"}, method="POST"),
                timeout=120,
            ).read(),
            daemon=True,
        )
        gen_thread.start()
        time.sleep(4)  # 让生成跑起来（mock 已产出几个 token）

        st = json_req(f"{base}/api/status", method="GET")
        print(f"[before stop] generating={st['generating']}")

        t0 = time.time()
        try:
            json_req(f"{base}/api/stop")
            print("[stop] ok")
        except Exception as e:
            print(f"[stop] FAILED: {e}")
            return

        # 观察 generating 是否在 3 秒内释放
        for i in range(15):
            time.sleep(0.2)
            st = json_req(f"{base}/api/status", method="GET")
            if not st["generating"]:
                print(f"[after stop] generating released after {time.time()-t0:.1f}s ✓")
                return
        st = json_req(f"{base}/api/status", method="GET")
        print(f"[after stop] STILL generating after 3s ✗ (卡死复现!) generating={st['generating']}")
    finally:
        HALT.set()
        mock.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()

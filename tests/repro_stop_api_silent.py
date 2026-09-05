"""API 后端「远端静默」停止回归测试。

真实 API 服务在思考/缓冲阶段可能长时间不下发任何数据。此时 server.py 的
_stream 阻塞在 backend.generate_stream 的 iter_lines 读上——若 stop() 的
resp.close() 不能打断另一线程的阻塞 recv（Windows 上的已知限制），生成锁
将永久不释放（卡死复现）。本测试模拟该场景并验证停止能及时生效。
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_PORT = 9102
SERVER_PORT = 9103
STOP_LATENCY_MAX = 8.0  # 停止后允许的最大收尾时间（秒）


class SilentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 真实 SSE 服务均为 HTTP/1.1 + chunked

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # 下发一个块后静默：模拟远端长时间不产出（思考/缓冲）→ 客户端读阻塞
        chunk = {
            "id": "m", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": {"content": "t0"}, "finish_reason": None}],
        }
        body = f"data: {json.dumps(chunk)}\n\n".encode()
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()
        try:
            while True:
                time.sleep(3600)  # 保持连接但永不发数据
        except Exception:
            pass

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
    mock = ThreadingHTTPServer(("127.0.0.1", MOCK_PORT), SilentHandler)
    threading.Thread(target=mock.serve_forever, daemon=True).start()

    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", str(SERVER_PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base = f"http://127.0.0.1:{SERVER_PORT}"
        for _ in range(50):
            try:
                json_req(f"{base}/api/status", method="GET")
                break
            except Exception:
                time.sleep(0.2)

        json_req(f"{base}/api/load", {
            "mode": "api", "base_url": f"http://127.0.0.1:{MOCK_PORT}/v1", "model": "m",
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
        time.sleep(2.5)  # 让流到达「静默」状态（读阻塞中）

        st = json_req(f"{base}/api/status", method="GET")
        print(f"[before stop] generating={st['generating']}")

        t0 = time.time()
        try:
            json_req(f"{base}/api/stop")
            print("[stop] ok")
        except Exception as e:
            print(f"[stop] FAILED: {e}")
            sys.exit(1)

        for i in range(int(STOP_LATENCY_MAX / 0.2) + 1):
            time.sleep(0.2)
            st = json_req(f"{base}/api/status", method="GET")
            if not st["generating"]:
                print(f"[after stop] generating released after {time.time()-t0:.1f}s OK")
                return
        st = json_req(f"{base}/api/status", method="GET")
        print(f"[after stop] STILL generating after {STOP_LATENCY_MAX}s - DEADLOCK generating={st['generating']}")
        sys.exit(1)
    finally:
        mock.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()

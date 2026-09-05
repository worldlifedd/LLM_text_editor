"""最小验证：Windows 上线程 B shutdown socket 能否打断线程 A 的阻塞 recv。"""
import socket
import threading
import time

s1, s2 = socket.socketpair()
blocked = threading.Event()
result = {}


def reader():
    try:
        result["read"] = s1.recv(4096)
    except Exception as e:  # noqa: BLE001
        result["err"] = f"{type(e).__name__}: {e}"
    blocked.set()


threading.Thread(target=reader, daemon=True).start()
time.sleep(0.5)  # 确保 recv 已阻塞

t0 = time.time()
try:
    s1.shutdown(socket.SHUT_RDWR)
    result["shutdown"] = "ok"
except Exception as e:  # noqa: BLE001
    result["shutdown"] = f"{type(e).__name__}: {e}"

blocked.wait(timeout=3)
print(f"[reader] after {time.time()-t0:.2f}s:", result, flush=True)
print("reader blocked?", not blocked.is_set(), flush=True)

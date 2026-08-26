# -*- coding: utf-8 -*-
"""模拟「未安装 torch」环境启动 server，验证 API-only 模式可无 torch 运行。

通过 meta_path 拦截 torch/transformers 的导入（模拟未安装），然后正常
启动 server —— 若 backend 顶层仍强依赖 torch，此处 import server 即会
报 ModuleNotFoundError，与用户在真机上的报错一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers"):
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, _BlockTorch())

import uvicorn  # noqa: E402
import server  # noqa: E402  无 torch 下加载 server 全部依赖

print("[no-torch] server 导入成功（无 torch），启动 :8912 ...", flush=True)
uvicorn.run(server.app, host="127.0.0.1", port=8912, log_level="warning")

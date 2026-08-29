# -*- coding: utf-8 -*-
"""llama-cpp-python 崩溃探针：verbose 加载，定位 0xc000001d 非法指令发生位置。"""
import sys

import llama_cpp

print("llama_cpp", llama_cpp.__version__, flush=True)
print("backend init ...", flush=True)
llama_cpp.llama_backend_init()
print("gpu offload supported:", llama_cpp.llama_supports_gpu_offload(), flush=True)
print("device count:", llama_cpp.llama_max_devices(), flush=True)

model = sys.argv[1] if len(sys.argv) > 1 else \
    r"E:\myGithub\LLM_text_editor\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
n_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"loading {model} n_gpu_layers={n_gpu} ...", flush=True)
llm = llama_cpp.Llama(model_path=model, n_ctx=512, n_gpu_layers=n_gpu, verbose=True)
print("LOAD OK, n_ctx =", llm.n_ctx, flush=True)

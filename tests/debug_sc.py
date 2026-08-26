# -*- coding: utf-8 -*-
"""最小复现：StoppingCriteria 是否被调用、scores 形态。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

calls = []


class Probe(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        calls.append((input_ids.shape, type(scores).__name__,
                      tuple(s.shape for s in scores) if isinstance(scores, (tuple, list)) else scores.shape))
        return torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
enc = tok("你好，", return_tensors="pt").to(model.device)
with torch.no_grad():
    model.generate(**enc, max_new_tokens=8, do_sample=False,
                   stopping_criteria=StoppingCriteriaList([Probe()]))
print("call count:", len(calls))
for c in calls[:4]:
    print(" ", c)

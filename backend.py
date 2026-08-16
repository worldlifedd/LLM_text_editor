# -*- coding: utf-8 -*-
"""LLM 后端：模型加载、流式生成、逐 token 困惑度采集、优雅停止。

两种后端，公共接口：loaded / load / build_chat_prompt / build_flat_prompt /
compute_context_ppl / generate_stream / stop / kind。
- LocalBackend：本地 transformers 模型（困惑度采集方案见下）。
- OpenAICompatBackend：OpenAI 兼容 Chat Completions API（OpenAI/DeepSeek/
  Qwen/GLM/Kimi/vLLM/Ollama 等），SSE 流式，logprobs 可用时计算逐 token 困惑度。

本地困惑度采集方案（transformers 4.4x 实测约束：stopping_criteria 收到的 scores
恒为 None/空，不可用）：
- _PplProcessor(LogitsProcessor)：每步拿到当前步（经 repetition_penalty 等
  处理后、采样 warper 前）的完整分布，log_softmax 后保留；下一步用
  input_ids[-1] 回填上一步实际选中 token 的 log-prob（滞后一步）。
- _TokenStream(TextIteratorStreamer)：put() 时记录全部生成 token id。
- 流结束时用保留的最后分布补齐尾部 token 的 log-prob。
显存开销：仅常驻一个 vocab 维向量。
"""
import json
import math
import threading
from dataclasses import dataclass, field

import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


def _common_prefix_len(a, b):
    """两个 token id 序列的最长公共前缀长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_MIN_CACHE_REUSE = 16  # 公共前缀低于此 token 数不值得复用缓存


@dataclass
class GenUpdate:
    """一次流式生成的增量快照。"""

    cum_text: str = ""                                   # 截至当前的累计生成文本
    token_ppls: list = field(default_factory=list)       # 每个 token 的困惑度 exp(-logprob)
    token_texts: list = field(default_factory=list)      # 与 ppl 对齐的文本片段（增量解码差分）
    final: bool = False                                  # 是否为本次流式的最后一次更新


class _PplProcessor(LogitsProcessor):
    """逐 token 困惑度采集（滞后一步回填）。

    注意：用户 logits_processor 在 _get_logits_processor 中于采样 warper
    （temperature/top_k/top_p）之前合并，因此这里的分布是「经惩罚项整形后的
    模型预测分布」——贪心模式下即最终决策分布；采样模式下为模型自身预测
    （未截断），更贴近"模型对文本的意外程度"语义。
    """

    def __init__(self):
        self.log_probs: list = []          # 每个 token 的 log-prob（滞后一步）
        self.prev_logprobs = None          # 上一步分布的 log_softmax（vocab,）

    def __call__(self, input_ids, scores):
        if self.prev_logprobs is not None:
            self.log_probs.append(self.prev_logprobs[input_ids[0, -1]].item())
        self.prev_logprobs = torch.log_softmax(scores[0].float(), dim=-1)
        return scores


class _TokenStream(TextIteratorStreamer):
    """流式输出同时记录生成 token id 序列（跳过首次 put 的 prompt tokens）。"""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_ids: list = []
        self._first_put = True

    def put(self, value):
        if self._first_put:
            # generate 会先 streamer.put(prompt_ids)（skip_prompt 机制依赖此行为）
            self._first_put = False
        else:
            self.token_ids.extend(value.reshape(-1).tolist())
        super().put(value)


class _StopOnEvent(StoppingCriteria):
    """优雅停止：外部 threading.Event 置位后在下一步返回 True。"""

    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return bool(self.stop_event.is_set())


class LocalBackend:
    """本地 transformers 模型封装（含 KV 前缀缓存加速 prefill）。"""

    kind = "local"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_name = ""
        self.device = "cpu"
        self._stop_event = threading.Event()
        # KV 前缀缓存：保存上次生成结束时的 KV（保守少记最后 1 个 token，
        # 因 generate 未必对最后一个 token 做前向），下次生成先求最长公共
        # 前缀，命中部分免 prefill，只对新增 token 增量前向。
        self._kv_cache: DynamicCache | None = None
        self._cache_ids: list = []
        self.last_cache_info = ""  # 最近一次生成的缓存命中信息（UI 展示用）
        self._gen_thread: threading.Thread | None = None  # 当前生成线程（串行化用）

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str):
        """加载本地路径或 HuggingFace hub 模型。CUDA -> fp16，CPU -> fp32。"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_path
        self._kv_cache, self._cache_ids, self.last_cache_info = None, [], ""

    # -------------------------------------------------------------- 聊天模板
    def apply_chat_template(self, messages, active_text):
        """messages → 模板化生成前缀文本（含 assistant 起始标记）+ 活动块续写头。

        提示词块映射为 user 指令、生成块为 assistant 回复，使 Instruct 模型
        正确区分"指令"与"正文"，避免把提示词当正文续写。
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt + (active_text or "")

    # ----------------------------------------------------------- prompt 构造
    def build_chat_prompt(self, messages, active_text):
        """chat 模式：本地后端 → 应用聊天模板后的平文本 str。"""
        return self.apply_chat_template(messages, active_text)

    def build_flat_prompt(self, flat_text):
        """prefix/raw 模式：本地后端 → 原样平文本 str。"""
        return flat_text

    # -------------------------------------------------------------- 困惑度
    @torch.no_grad()
    def compute_context_ppl(self, text: str) -> float:
        """上下文困惑度：一次前向传播 exp(mean CE)。提示词准确度指标。"""
        if not self.loaded or not text.strip():
            return float("nan")
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if enc.input_ids.shape[1] < 2:
            return float("nan")
        out = self.model(**enc, labels=enc.input_ids)
        return math.exp(min(out.loss.item(), 20.0))

    # ---------------------------------------------------------------- 生成
    def stop(self):
        """请求优雅停止：StoppingCriteria 返回 True，generate 自然收尾。"""
        self._stop_event.set()

    def generate_stream(self, context: str, **gen_params):
        """流式生成。yield GenUpdate；自然结束/停止后额外 yield final=True 快照。

        gen_params: max_new_tokens / do_sample / temperature / top_k / top_p /
        repetition_penalty。
        """
        if not self.loaded:
            raise RuntimeError("模型尚未加载，请先在顶栏加载模型")

        # 串行化：若上一次生成线程尚未退出（如刚点停止就重启生成），
        # 先请求其收尾并等待退出——两个 generate 并发跑会同时写同一个
        # KV cache 对象（crop/append 竞态），导致 cache 状态错乱。
        if self._gen_thread is not None and self._gen_thread.is_alive():
            self._stop_event.set()
            self._gen_thread.join(timeout=30)
            if self._gen_thread.is_alive():
                raise RuntimeError("上一次生成仍在运行，请稍后重试")

        self._stop_event = threading.Event()
        self._gen_error: Exception | None = None
        streamer = _TokenStream(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        proc = _PplProcessor()

        # ---- prefill：KV 前缀缓存命中部分免重算，仅对新增 token 增量前向 ----
        # 4.49 generate 语义：传完整 input_ids + past_key_values（含前 n-1 个
        # token 的 KV），prepare_inputs_for_generation 自动只喂最后一个 token。
        ids = self.tokenizer(context, return_tensors=None)["input_ids"]
        if len(ids) < 2:
            self._kv_cache, self._cache_ids = None, []
        cache, k = self._kv_cache, 0
        if cache is not None:
            k = _common_prefix_len(self._cache_ids, ids)
            if k < _MIN_CACHE_REUSE:
                cache, k = None, 0
        if cache is None:
            cache = DynamicCache()
        else:
            if k >= len(ids):
                # 缓存已覆盖整个 prompt（如立即重新生成同一上下文）：
                # 回退 1 个 token 作为 generate 的种子，否则 generate 的
                # cache_position 切片为空会 IndexError
                k = len(ids) - 1
            cache.crop(k)
        prefill_ids = ids[k:-1]  # 缓存未覆盖且非末位（末位作为 generate 种子）
        if prefill_ids:
            dev = self.model.device
            try:
                with torch.no_grad():
                    self.model(
                        input_ids=torch.tensor([prefill_ids], dtype=torch.long, device=dev),
                        attention_mask=torch.ones(1, k + len(prefill_ids), dtype=torch.long, device=dev),
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.arange(k, k + len(prefill_ids), device=dev),
                    )
            except Exception:  # noqa: BLE001  预填失败（如显存不足）→ 弃缓存全量重来
                self._kv_cache, self._cache_ids, cache, k = None, [], None, 0
                cache = DynamicCache()
                prefill_ids = ids[:-1]
                with torch.no_grad():
                    self.model(
                        input_ids=torch.tensor([prefill_ids], dtype=torch.long, device=self.model.device),
                        attention_mask=torch.ones(1, len(prefill_ids), dtype=torch.long, device=self.model.device),
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.arange(0, len(prefill_ids), device=self.model.device),
                    )
        self.last_cache_info = (
            f"⚡KV缓存复用 {k}/{len(ids)} token"
            if k
            else "KV缓存未命中"
        )
        self._kv_cache, self._cache_ids = cache, ids[:-1]

        enc_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        do_sample = bool(gen_params.get("do_sample", True))
        kwargs = dict(
            max_new_tokens=int(gen_params.get("max_new_tokens", 256)),
            do_sample=do_sample,
            repetition_penalty=float(gen_params.get("repetition_penalty", 1.1)),
            streamer=streamer,
            past_key_values=cache,
            use_cache=True,
            logits_processor=LogitsProcessorList([proc]),
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(self._stop_event)]),
        )
        if do_sample:
            kwargs.update(
                temperature=float(gen_params.get("temperature", 0.8)),
                top_k=int(gen_params.get("top_k", 50)),
                top_p=float(gen_params.get("top_p", 0.95)),
            )

        def _run():
            try:
                with torch.no_grad():
                    self.model.generate(
                        input_ids=enc_ids,
                        attention_mask=torch.ones(1, len(ids), dtype=torch.long,
                                                   device=self.model.device),
                        **kwargs,
                    )
            except Exception as e:  # noqa: BLE001
                # generate 线程崩溃时必须 end() 解除主线程的流式迭代阻塞，
                # 并记录错误供 generate_stream 主循环抛出
                self._gen_error = e
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        self._gen_thread = thread
        thread.start()

        # token-文本对齐（增量解码差分法）：逐 token decode 前缀取差分，
        # 片段与 ppl 一一对应，中文多字节字符跨 token 不乱码。
        aligned_texts: list = []
        aligned_len = 0
        seen = 0
        cum_text = ""

        def _pull_new_tokens():
            nonlocal seen, aligned_len
            # log_probs 滞后一步：仅对齐已有 ppl 的 token
            n_avail = min(len(streamer.token_ids), len(proc.log_probs))
            for i in range(seen, n_avail):
                prefix_text = self.tokenizer.decode(
                    streamer.token_ids[: i + 1], skip_special_tokens=True
                )
                diff = prefix_text[aligned_len:]
                aligned_texts.append(diff)
                aligned_len += len(diff)
                seen = i + 1

        try:
            for chunk in streamer:
                if chunk:
                    cum_text += chunk
                _pull_new_tokens()
                ppls = [math.exp(min(-lp, 20.0)) for lp in proc.log_probs[:seen]]
                yield GenUpdate(
                    cum_text=cum_text, token_ppls=ppls, token_texts=list(aligned_texts)
                )
        finally:
            # 收尾：join 线程；用保留的最后分布补齐尾部 token 的 log-prob
            thread.join(timeout=30)
            if self._gen_error is not None:
                self._kv_cache = None  # 缓存状态未知，弃用
                raise self._gen_error
            if thread.is_alive():
                # 线程未如期退出（极端卡顿），缓存正在被并发写入 → 弃用，
                # 宁可下次全量 prefill 也不冒竞态风险
                self._kv_cache = None
            if (
                proc.prev_logprobs is not None
                and len(proc.log_probs) < len(streamer.token_ids)
            ):
                last_tok = streamer.token_ids[len(proc.log_probs)]
                proc.log_probs.append(proc.prev_logprobs[last_tok].item())
                proc.prev_logprobs = None
            _pull_new_tokens()
            _final_ppls = [math.exp(min(-lp, 20.0)) for lp in proc.log_probs[:seen]]
            _final_snapshot = GenUpdate(
                cum_text=cum_text,
                token_ppls=_final_ppls,
                token_texts=aligned_texts,
                final=True,
            )
            # 保存 KV 缓存供下次复用：完整序列 = prompt + 生成。保守裁掉
            # 最后 1 个 token（generate 未必对它做过前向），保证缓存状态
            # 与 _cache_ids 严格一致。
            if self._kv_cache is not None:
                full_ids = ids + streamer.token_ids
                safe_len = max(len(full_ids) - 1, 0)
                self._kv_cache.crop(safe_len)
                self._cache_ids = full_ids[:safe_len]
        yield _final_snapshot


class OpenAICompatBackend:
    """OpenAI 兼容 Chat Completions API 后端。

    覆盖 OpenAI / DeepSeek / Qwen(DashScope compatible-mode) / GLM / Kimi /
    Mistral，以及 vLLM、Ollama、llama.cpp server 等本地推理服务。
    - 流式：SSE data: {...} 行解析，delta.content 累积为 cum_text。
    - 逐 token 困惑度：请求 logprobs=True；SUPPORTS 的服务端返回
      choices[0].logprobs.content[*]（token + logprob），据此计算 ppl。
      不支持的服务端（如 DeepSeek）首次 400 后自动去掉该参数降级。
    - 上下文困惑度：Chat Completions 不返回 prompt token 概率 → 返回 NaN。
    """

    kind = "api"

    def __init__(self):
        self.base_url = ""
        self.api_key = ""
        self.model = ""
        self.supports_logprobs = None  # None=未探测（首次生成时探测）
        self._stop_event = threading.Event()
        self._resp = None

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return bool(self.model and self.base_url)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def load(self, base_url: str, api_key: str, model: str):
        """配置并验证连接。base_url 填写到 /v1 层级（不含 /chat/completions）。"""
        base = (base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if not base:
            raise ValueError("base_url 不能为空，例如 https://api.openai.com/v1 或 http://localhost:11434/v1")
        if not (model or "").strip():
            raise ValueError("模型名不能为空，例如 gpt-4o-mini / deepseek-chat / qwen-plus")

        self.base_url, self.api_key, self.model = base, (api_key or "").strip(), model.strip()
        self.supports_logprobs = None

        # 连接验证：GET /models（401/403 = 鉴权失败；404/405 = 网关无此端点，放行）
        try:
            r = requests.get(f"{base}/models", headers=self._headers(), timeout=15)
        except requests.RequestException as e:
            self.base_url = ""
            raise ValueError(f"无法连接 {base}：{e}") from e
        if r.status_code in (401, 403):
            self.base_url = ""
            raise ValueError(f"API Key 无效或无权限（HTTP {r.status_code}）")
        # 404/405：部分网关不实现 /models，放行，生成时再验证

    # ----------------------------------------------------------- prompt 构造
    def build_chat_prompt(self, messages, active_text):
        """chat 模式：API 后端 → messages 列表。活动块文本作为末尾 assistant
        消息（prefill），模型以其为上文继续输出新内容。"""
        msgs = [dict(m) for m in messages]
        if (active_text or "").strip():
            msgs.append({"role": "assistant", "content": active_text})
        return msgs

    def build_flat_prompt(self, flat_text):
        """prefix/raw 模式：平文本包成单条 user 消息。"""
        return [{"role": "user", "content": flat_text}]

    # -------------------------------------------------------------- 困惑度
    def compute_context_ppl(self, prompt) -> float:
        """API 无法获取 prompt token 概率，上下文困惑度不可用。"""
        return float("nan")

    # ---------------------------------------------------------------- 生成
    def stop(self):
        self._stop_event.set()
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:  # noqa: BLE001
                pass

    def _post_stream(self, payload):
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url, headers=self._headers(), json=payload, stream=True, timeout=(15, 600)
        )
        if resp.status_code != 200 and "logprobs" in payload and resp.status_code == 400:
            # 服务端不支持 logprobs 参数 → 去掉后重试一次并记住
            resp.close()
            self.supports_logprobs = False
            payload.pop("logprobs")
            resp = requests.post(
                url, headers=self._headers(), json=payload, stream=True, timeout=(15, 600)
            )
        if resp.status_code != 200:
            body = resp.text[:300]
            resp.close()
            raise RuntimeError(f"API 请求失败（HTTP {resp.status_code}）：{body}")
        resp.encoding = "utf-8"  # SSE 未带 charset 时 requests 默认 latin-1，中文会乱码
        return resp

    def generate_stream(self, prompt, **gen_params):
        """流式生成。prompt 为 messages 列表；yield GenUpdate。"""
        if not self.loaded:
            raise RuntimeError("API 未连接，请先在顶栏配置并连接")

        self._stop_event = threading.Event()
        do_sample = bool(gen_params.get("do_sample", True))
        payload = dict(
            model=self.model,
            messages=prompt,
            stream=True,
            max_tokens=int(gen_params.get("max_new_tokens", 256)),
            temperature=1e-8 if not do_sample else float(gen_params.get("temperature", 0.8)),
            top_p=float(gen_params.get("top_p", 0.95)),
        )
        if self.supports_logprobs is not False:
            payload["logprobs"] = True

        resp = self._post_stream(payload)
        self._resp = resp
        if "logprobs" in payload and self.supports_logprobs is None:
            self.supports_logprobs = True  # 200 通过，待首条数据确认

        cum_text = ""
        token_texts: list = []
        token_ppls: list = []
        final_snapshot = None
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if self._stop_event.is_set():
                    break
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    cum_text += piece
                # logprobs 逐 token 困惑度（服务端支持时）
                for ent in (choice.get("logprobs") or {}).get("content") or []:
                    tok = ent.get("token") or ""
                    logprob = ent.get("logprob")
                    if tok and logprob is not None:
                        token_texts.append(tok)
                        token_ppls.append(math.exp(min(-logprob, 20.0)))
                yield GenUpdate(
                    cum_text=cum_text, token_ppls=list(token_ppls),
                    token_texts=list(token_texts),
                )
        except (ValueError, requests.RequestException):
            # stop() 关闭连接导致迭代异常 → 视为正常停止
            if not self._stop_event.is_set():
                raise
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            self._resp = None
            if self.supports_logprobs and not token_ppls:
                self.supports_logprobs = False  # 200 但从未返回 logprobs 数据
            final_snapshot = GenUpdate(
                cum_text=cum_text, token_ppls=token_ppls,
                token_texts=token_texts, final=True,
            )
        yield final_snapshot

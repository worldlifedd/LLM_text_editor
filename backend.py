# -*- coding: utf-8 -*-
"""LLM 后端：模型加载、流式生成、逐 token 困惑度采集、优雅停止。

三种后端，公共接口：loaded / load / build_chat_prompt / build_flat_prompt /
compute_context_ppl / generate_stream / stop / kind。
- LocalBackend：本地 transformers 模型（困惑度采集方案见下）。
- LlamaCppBackend：llama-cpp-python 进程内加载 GGUF 量化模型
  （Q4_K_M / Q5_K_M / Q8_0 / IQ 系列等），适配各种量化 LLM 部署。
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

torch/transformers 与 llama_cpp 均为惰性加载：仅对应后端真正使用时才引入，
因此只用 API 模式时无需安装 torch / llama-cpp-python（见 _local_deps /
_require_llama_cpp）。
"""
import json
import math
import os
import queue
import sys
import threading
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------- 惰性加载
_torch = None
_tf = None


def _require_torch():
    """惰性引入 torch/transformers，返回 (torch, transformers)。"""
    global _torch, _tf
    if _torch is None:
        import torch
        import transformers

        _torch, _tf = torch, transformers
    return _torch, _tf


_LOCAL_DEPS = None


def _local_deps():
    """构造并缓存本地模式所需的 torch/transformers 依赖（含子类定义）。

    子类需在 transformers 导入后才能定义，故统一放进这里懒加载，
    避免模块导入期强依赖 torch（API-only 安装可无 torch）。
    """
    global _LOCAL_DEPS
    if _LOCAL_DEPS is not None:
        return _LOCAL_DEPS
    torch, tf = _require_torch()

    class _PplProcessor(tf.LogitsProcessor):
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

    class _TokenStream(tf.TextIteratorStreamer):
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

    class _StopOnEvent(tf.StoppingCriteria):
        """优雅停止：外部 threading.Event 置位后在下一步返回 True。"""

        def __init__(self, stop_event: threading.Event):
            super().__init__()
            self.stop_event = stop_event

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return bool(self.stop_event.is_set())

    _LOCAL_DEPS = {
        "torch": torch,
        "AutoModelForCausalLM": tf.AutoModelForCausalLM,
        "AutoTokenizer": tf.AutoTokenizer,
        "DynamicCache": tf.DynamicCache,
        "LogitsProcessorList": tf.LogitsProcessorList,
        "StoppingCriteriaList": tf.StoppingCriteriaList,
        "_PplProcessor": _PplProcessor,
        "_TokenStream": _TokenStream,
        "_StopOnEvent": _StopOnEvent,
    }
    return _LOCAL_DEPS


_llama_cpp = None


def _require_llama_cpp():
    """惰性引入 llama_cpp（llama-cpp-python），未安装时给出安装指引。"""
    global _llama_cpp
    if _llama_cpp is None:
        try:
            import llama_cpp
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python 未安装：pip install llama-cpp-python"
                "（CUDA 加速版安装方法见"
                " https://github.com/abetlen/llama-cpp-python#usage-with-gpu）"
            ) from e
        _llama_cpp = llama_cpp
    return _llama_cpp


def _common_prefix_len(a, b):
    """两个 token id 序列的最长公共前缀长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_MIN_CACHE_REUSE = 16  # 公共前缀低于此 token 数不值得复用缓存

# 思维链标签（Qwen3 / DeepSeek-R1 系本地模型内嵌输出）。
# 拼接构造以避免标签字面量被工具链按 HTML 清洗。
_THINK_OPEN = "<" + "thi" + "nk>"
_THINK_CLOSE = "</" + "thi" + "nk>"


class _ThinkSplitter:
    """流式思维链分离器：把 <think>…</think> 文本流切成（思维链, 正文）。

    两种起始形态（推理模型约定思考区在输出最前）：
    - 开标签在生成流内：如 DeepSeek-R1 / Qwen3 默认输出，流首即 <think>；
      初始 probe 态先判定是否命中开标签，未命中则整段视为正文。
    - 开标签已在 prompt 内：Qwen3 系模板在 enable_thinking=true 时渲染出的
      prompt 以未闭合的 <think> 结尾，生成流直接从思考正文开始、只含闭标签；
      此时须以 assume_thinking=True 构造，否则思考内容会被整段误判为正文，
      且闭标签会原样泄漏进正文。

    think 态保留可能是右标签前缀的尾部；content_start() 给出解码坐标下的
    正文起点，供 token 对齐时剔除思考区 token。
    """

    def __init__(self, assume_thinking: bool = False):
        # assume_thinking：prompt 已带开标签，流首即思考正文（跳过 probe 判定）
        self.state = "think" if assume_thinking else "probe"
        self._open_len = 0 if assume_thinking else len(_THINK_OPEN)
        self._pending = ""        # probe 未决文本 / think 中右标签的候选前缀
        self._fed = 0             # 已喂入总字符数（解码坐标）
        self._content_start = 0   # 正文起点（仅 content 态有效）
        self.closed = False       # 流中是否出现过思考区闭标签（供前端补写/冻结）

    def content_start(self):
        """非正文前缀长度（解码坐标）；非 content 态返回 None（暂无正文）。"""
        if self.state != "content":
            return None
        return self._content_start

    def reasoning_start(self):
        """思考区正文左端点（解码坐标）：开标签之后（开标签在 prompt 内的
        形态 B 为 0）。供 token 对齐按区间切片，剔除标签字符。"""
        return self._open_len

    def fed_len(self):
        """已喂入文本总长度（解码坐标）：token 对齐的消费上界——生成线程
        可能已把后续 token 写进 token_ids，但其文本 chunk 尚未被消费喂入
        分离器（线程先于消费推进），越界即暂缓定性，保证 join(token_texts)
        恒为 cum_text 前缀（契约 C3）。"""
        return self._fed

    def feed(self, chunk):
        """喂入一段文本 → (思维链增量, 正文增量)。"""
        if not chunk:
            return "", ""
        self._fed += len(chunk)
        if self.state == "probe":
            self._pending += chunk
            if len(self._pending) < len(_THINK_OPEN) and _THINK_OPEN.startswith(self._pending):
                return "", ""  # 仍可能是开标签前缀：暂缓判定
            if self._pending.startswith(_THINK_OPEN):
                rest = self._pending[len(_THINK_OPEN):]
                self.state, self._pending = "think", ""
                return self._think(rest)
            self.state, self._content_start = "content", 0
            text, self._pending = self._pending, ""
            return "", text
        if self.state == "think":
            return self._think(chunk)
        return "", chunk

    def _think(self, chunk):
        self._pending += chunk
        i = self._pending.find(_THINK_CLOSE)
        if i < 0:
            # 保留可能为右标签前缀的尾部，其余即思维链
            keep = max(0, len(self._pending) - (len(_THINK_CLOSE) - 1))
            out, self._pending = self._pending[:keep], self._pending[keep:]
            return out, ""
        reasoning = self._pending[:i]
        rest = self._pending[i + len(_THINK_CLOSE):]
        # 开标签在 prompt 内时它不占流内坐标，故计入 _open_len（此时为 0）
        self._content_start = self._open_len + len(reasoning) + len(_THINK_CLOSE)
        self.state, self._pending = "content", ""
        self.closed = True   # 思考区闭合：前端据此在 cot 块末尾补写闭标签
        return reasoning, rest

    def flush(self):
        """流结束：未决文本归位（probe→正文；think 尾部→思维链）。"""
        if self.state == "probe":
            self.state, self._content_start = "content", 0
            text, self._pending = self._pending, ""
            return "", text
        if self.state == "think":
            out, self._pending = self._pending, ""
            return out, ""
        return "", ""


def _prompt_pending_think(prompt) -> bool:
    """prompt 是否处于未闭合的思考区（Qwen3 系模板 enable_thinking=true）。

    此时生成流首即思考正文、只含闭标签，分离器须以 assume_thinking=True
    启动，否则思考内容会被当作正文、闭标签还会泄漏进正文。

    全文扫描而非固定尾部窗口：续写场景下回灌的思维链可能有上千字符，
    开标签会被挤出任何固定大小的窗口，导致误判为"不在思考区"。
    """
    if not isinstance(prompt, str) or not prompt:
        return False
    i = prompt.rfind(_THINK_OPEN)
    if i < 0:
        return False
    return prompt.find(_THINK_CLOSE, i) < 0


def _splice_pending_think(prompt: str, cot_prefix: str) -> str:
    """把文档中的思维链作为思考区回灌，行为由 cot 块的闭合状态决定（所见即所得）。

    插件端把 `<think>` 与 `<<arg_key:6124c78e>>` 直接写在 cot 注释块里，用户
    可编辑：保留闭标签 → 思考已完成，模型直接给正文；删掉闭标签 → 思考被中断，
    模型从断点继续思考（重新实测 Qwen3.5-4B：未闭合时模型会一直思考不闭合，
    这正是"接续思考"的行为，用户可随时定稿或补闭标签结束）。

    两种 cot 形态：
    - 新格式（cot 以 `<think>` 开头）：自包含完整思考区，直接替换模板渲染的
      思考区（模板可能渲染出未闭合的 `<think>` 或闭合的空块）。
    - 旧格式（纯文本，历史文档）：按闭合回灌处理（补全标签），保证可读性。
    """
    if not cot_prefix:
        return prompt
    if not isinstance(prompt, str) or not prompt:
        return cot_prefix if cot_prefix.strip().startswith(_THINK_OPEN) else (
            _THINK_OPEN + "\n" + cot_prefix + "\n" + _THINK_CLOSE + "\n\n"
        )
    i = prompt.rfind(_THINK_OPEN)
    if i < 0:
        # 模板未渲染思考区标记：原样接在末尾
        return prompt + "\n" + cot_prefix
    j = prompt.find(_THINK_CLOSE, i)
    rest = prompt[j + len(_THINK_CLOSE):] if j >= 0 else ""
    if cot_prefix.strip().startswith(_THINK_OPEN):
        # 新格式：cot 自带标签。提取正文并重建思考区，**保持模板渲染的
        # "<think>\n" 结构**——若把 cot 的 <think> 原样替换进来（其后再接
        # 正文而非换行），token 序列在 <think> 后即与上次生成的
        # "<think>\n…" 前缀分叉（实测只匹配到模板部分 22/322），KV 缓存
        # 复用失效 → 回灌的整段思维链全量 prefill，随长度线性变慢。
        s = cot_prefix.strip()
        body = s[len(_THINK_OPEN):]          # 去掉开标签
        closed = False
        ci = body.rfind(_THINK_CLOSE)
        if ci >= 0:
            body = body[:ci]
            closed = True
        body = body.strip("\n")              # 模板思考区自带 \n 分隔
        if closed:
            return prompt[:i] + _THINK_OPEN + "\n" + body + "\n" + _THINK_CLOSE + rest
        return prompt[:i] + _THINK_OPEN + "\n" + body + rest
    # 旧格式：纯文本 → 闭合回灌（模型稳定出正文）
    return prompt[:i] + _THINK_OPEN + "\n" + cot_prefix + "\n" + _THINK_CLOSE + rest


@dataclass
class GenUpdate:
    """一次流式生成的增量快照。

    三后端（local/llamacpp/api）统一的行为契约（前端 web/editor.js 与
    VSCode 插件 generation.ts 的着色对账都依赖，回归锚点见
    tests/test_backend_contract.py）：
    - final=True 的快照有且只有最后一个（自然结束/停止/出错后收尾）；
    - token_texts 与 token_ppls、reasoning_token_texts 与
      reasoning_token_ppls 一一等长；
    - join(token_texts) 恒为 cum_text 的前缀，final 时相等（logprobs
      不可用的后端/服务端则为空列表）；
    - join(reasoning_token_texts) 恒为 reasoning_cum 的前缀，final 时相等，
      且不含思考区标签字符（仅思考正文）；API 后端无 reasoning
      logprobs → 恒为空（思维链不着色是已知的降级，非失配）。
    """

    cum_text: str = ""                                   # 截至当前的累计生成文本（正文，不含思维链）
    reasoning_cum: str = ""                              # 截至当前的累计思维链（推理模型）
    token_ppls: list = field(default_factory=list)       # 每个 token 的困惑度 exp(-logprob)
    token_texts: list = field(default_factory=list)      # 与 ppl 对齐的文本片段（增量解码差分）
    reasoning_token_ppls: list = field(default_factory=list)  # 思考区 token 困惑度（本地后端）
    reasoning_token_texts: list = field(default_factory=list) # 与其对齐的文本片段
    reasoning_closed: bool = False                       # 思考区是否已闭合（模型输出过闭标签）
    final: bool = False                                  # 是否为本次流式的最后一次更新


# 推理模型名启发式（思维链支持静态推断的弱信号）
_REASONING_NAME_HINTS = ("r1", "qwq", "reason", "thinking", "gpt-o", "glm-z", "glm4-z")

# 思维链支持探测结果：yes=确定支持 no=确定不支持 unknown=待运行时检测
ReasoningSupport = dict


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
        self._kv_cache = None  # 运行时为 transformers DynamicCache | None
        self._cache_ids: list = []
        self.last_cache_info = ""  # 最近一次生成的缓存命中信息（UI 展示用）
        self._gen_thread: threading.Thread | None = None  # 当前生成线程（串行化用）
        self.reasoning_seen = False  # 运行时检测：本模型曾输出思维链

    # ------------------------------------------------------- 思维链能力探测
    def _chat_template_text(self) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(tpl, str) or not tpl:
            # chat_template 可能是模板列表（多模板），拼起来检查
            tpls = getattr(self.tokenizer, "chat_template", None)
            return json.dumps(tpls, ensure_ascii=False) if tpls else ""
        return tpl

    def reasoning_support(self) -> ReasoningSupport:
        """思维链能力：supported=yes/no/unknown；toggleable=模板支持开关。

        静态信号：聊天模板含 enable_thinking 变量（Qwen3 系，可开关）或
        模型名命中推理系关键词；unknown 则留待运行时检测（输出含
        <think> 标签即确认）。
        """
        if not self.loaded:
            return {"supported": "unknown", "toggleable": False}
        tpl = self._chat_template_text()
        toggleable = "enable_thinking" in tpl
        if self.reasoning_seen:
            return {"supported": "yes", "toggleable": toggleable}
        name = (self.model_name or "").lower()
        hinted = any(k in name for k in _REASONING_NAME_HINTS)
        if hinted or toggleable:
            return {"supported": "yes", "toggleable": toggleable}
        # R1-Distill 类：模板无开关但输出内嵌 <think>；无法静态确定
        return {"supported": "unknown", "toggleable": False}

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def unload(self):
        """释放模型、KV 缓存并归还显存（切换后端时由服务端调用）。"""
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self._kv_cache = None
        self._cache_ids = []
        self.last_cache_info = ""
        self.reasoning_seen = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001  torch 不可用时仅丢弃引用
            pass

    def load(self, model_path: str):
        """加载本地路径或 HuggingFace hub 模型。CUDA -> fp16，CPU -> fp32。"""
        d = _local_deps()
        torch = d["torch"]
        self.tokenizer = d["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = d["AutoModelForCausalLM"].from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_path
        self._kv_cache, self._cache_ids, self.last_cache_info = None, [], ""
        self.reasoning_seen = False

    # -------------------------------------------------------------- 聊天模板
    def apply_chat_template(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        """messages → 模板化生成前缀文本（含 assistant 起始标记）+ 活动块续写头。

        提示词块映射为 user 指令、生成块为 assistant 回复，使 Instruct 模型
        正确区分"指令"与"正文"，避免把提示词当正文续写。
        enable_thinking：思考模式开关（仅模板含该变量时生效，如 Qwen3）；
        None=跟随模型默认。
        cot_prefix：被中断的思维链（正文尚未开始时的续写头），拼进思考区。
        """
        kwargs = {}
        if enable_thinking is not None and "enable_thinking" in self._chat_template_text():
            kwargs["enable_thinking"] = bool(enable_thinking)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        return _splice_pending_think(prompt, cot_prefix) + (active_text or "")

    # ----------------------------------------------------------- prompt 构造
    def build_chat_prompt(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        """chat 模式：本地后端 → 应用聊天模板后的平文本 str。"""
        return self.apply_chat_template(messages, active_text, enable_thinking, cot_prefix)

    def build_flat_prompt(self, flat_text):
        """prefix/raw 模式：本地后端 → 原样平文本 str。"""
        return flat_text

    # -------------------------------------------------------------- 困惑度
    def compute_context_ppl(self, text: str) -> float:
        """上下文困惑度：一次前向传播 exp(mean CE)。提示词准确度指标。"""
        return self.score_context(text)[0]

    def score_context(self, context: str, tail_text: str = "", abort_event=None):
        """上下文打分（单次前向）：整段困惑度 + 尾部文本逐 token 困惑度。

        prefill 本就要前向整个上下文，故同一趟同时产出 prompt 困惑度与
        尾部文本（活动块内手动编辑的部分）的逐 token ppl，前端据此为
        编辑文本着色。返回 (ctx_ppl, tail_token_texts, tail_token_ppls)；
        ctx_ppl 不可用时为 nan，尾部无法按 token 边界对齐时为空列表。

        abort_event：可选停止信号。命中（停止请求在打分开始前已到达）时
        跳过整趟前向直接返回不可用——避免无用 prefill 拖住停止生效。
        （单次前向本身不可中断；打分开始后到达的停止由调用方兜底。）
        """
        if not self.loaded or not context.strip():
            return float("nan"), [], []
        if abort_event is not None and abort_event.is_set():
            return float("nan"), [], []
        torch = _require_torch()[0]
        enc = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        if enc.input_ids.shape[1] < 2:
            return float("nan"), [], []
        with torch.no_grad():
            out = self.model(**enc, labels=enc.input_ids)
        ctx = math.exp(min(out.loss.item(), 20.0))
        tail_t, tail_p = self._score_tail(
            context, enc.input_ids[0].tolist(), out.logits, tail_text
        )
        return ctx, tail_t, tail_p

    def _score_tail(self, context, ids, logits, tail_text):
        """尾部文本逐 token 打分：logits[i-1] 预测 ids[i] 的 log-prob。

        token 边界用「头部单独 tokenize 必须是全量 tokenize 的前缀」判定
        （模板尾部与活动文本在 BPE 层合并的罕见情形下放弃打分）；
        文本片段用增量解码差分法对齐（与 generate_stream 同语义）。
        """
        torch = _require_torch()[0]
        if not tail_text or not context.endswith(tail_text):
            return [], []
        head_ids = self.tokenizer(
            context[: len(context) - len(tail_text)], return_tensors=None
        )["input_ids"]
        start = len(head_ids)
        if start < 1 or start >= len(ids) or ids[:start] != head_ids:
            return [], []
        # 分块 log_softmax 控制大词表长尾的瞬时显存峰值
        lps: list = []
        lo = start
        while lo < len(ids):
            hi = min(lo + 128, len(ids))
            chunk = torch.log_softmax(logits[0, lo - 1: hi - 1].float(), dim=-1)
            tgt = torch.tensor(ids[lo:hi], dtype=torch.long, device=chunk.device)
            lps.extend(chunk.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist())
            lo = hi
        texts, ppls, base_len = [], [], len(
            self.tokenizer.decode(ids[:start], skip_special_tokens=True)
        )
        for i, lp in enumerate(lps, start=start):
            prefix = self.tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
            piece = prefix[base_len:]
            base_len = len(prefix)
            if not piece:
                continue
            texts.append(piece)
            ppls.append(math.exp(min(-lp, 20.0)))
        if not texts or not tail_text.endswith("".join(texts)):
            return [], []  # 对齐失配（尾部含特殊 token 等）→ 放弃
        return texts, ppls

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

        d = _local_deps()
        torch = d["torch"]

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
        streamer = d["_TokenStream"](
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        proc = d["_PplProcessor"]()

        # ---- prefill：KV 前缀缓存命中部分免重算，仅对新增 token 增量前向 ----
        # 4.49 generate 语义：传完整 input_ids + past_key_values（含前 n-1 个
        # token 的 KV），prepare_inputs_for_generation 自动只喂最后一个 token。
        ids = self.tokenizer(context, return_tensors=None)["input_ids"]
        if len(ids) < 2:
            self._kv_cache, self._cache_ids = None, []
        # 混合线性注意力模型（Qwen3.5 gated deltanet 等，config.layer_types
        # 标记 linear_attention / mamba* 层）在 transformers 5.x 须由 generate
        # 按 config 自建专用缓存：手动构造/传入 DynamicCache 会在
        # update_conv_state 处 IndexError（list index out of range），且无法
        # 手动 prefill。此类模型禁用手动 KV 缓存（全量前向交由 generate）。
        layer_types = getattr(self.model.config, "layer_types", None) or []
        hybrid_linear = any(
            isinstance(t, str) and ("linear" in t or "mamba" in t) for t in layer_types
        )
        if hybrid_linear:
            self._kv_cache, self._cache_ids = None, []
        cache, k = self._kv_cache, 0
        if cache is not None:
            k = _common_prefix_len(self._cache_ids, ids)
            if k < _MIN_CACHE_REUSE:
                cache, k = None, 0
        if cache is None and not hybrid_linear:
            cache = d["DynamicCache"]()
        elif cache is not None:
            if k >= len(ids):
                # 缓存已覆盖整个 prompt（如立即重新生成同一上下文）：
                # 回退 1 个 token 作为 generate 的种子，否则 generate 的
                # cache_position 切片为空会 IndexError
                k = len(ids) - 1
            cache.crop(k)
        prefill_ids = ids[k:-1] if cache is not None else []  # 末位作为 generate 种子
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
                cache = d["DynamicCache"]()
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
            else "KV缓存不可用（混合注意力模型）" if hybrid_linear
            else "KV缓存未命中"
        )
        self._kv_cache, self._cache_ids = (
            (cache, ids[:-1]) if cache is not None else (None, [])
        )

        enc_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        do_sample = bool(gen_params.get("do_sample", True))
        kwargs = dict(
            max_new_tokens=int(gen_params.get("max_new_tokens", 256)),
            do_sample=do_sample,
            repetition_penalty=float(gen_params.get("repetition_penalty", 1.1)),
            streamer=streamer,
            use_cache=True,
            logits_processor=d["LogitsProcessorList"]([proc]),
            stopping_criteria=d["StoppingCriteriaList"]([d["_StopOnEvent"](self._stop_event)]),
        )
        # 混合注意力模型不传 past_key_values：由 generate 按 config 自建
        # 专用缓存（传 DynamicCache 会 IndexError）
        if cache is not None:
            kwargs["past_key_values"] = cache
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
        # 思考区 token 的 ppl 与正文同源 → 思维链同样困惑度着色（r_aligned），
        # 但标签字符不入序列（join(r_aligned) == reasoning_cum，见下）。
        aligned: list = []   # (text, ppl) 仅正文 token
        r_aligned: list = [] # (text, ppl) 思考区正文 token（不含标签字符）
        aligned_len = 0
        seen = 0
        cum_text = ""
        reasoning_cum = ""
        # 模板已预置 <think> 时流首即思考正文，分离器需直接进入 think 态
        splitter = _ThinkSplitter(assume_thinking=_prompt_pending_think(context))

        def _ppl_of(i):
            return math.exp(min(-proc.log_probs[i], 20.0))

        def _pull_new_tokens():
            nonlocal seen, aligned_len
            # probe 态标签未定，暂不消费；think/content 态按坐标区间切片：
            #   思考区正文 [r0, r1) → r_aligned；正文 [head, ∞) → aligned；
            #   开/闭标签字符两不收——join(r_aligned) 必须与 reasoning_cum
            #   逐字相等（前端/插件按前缀覆盖对账，混入标签即整体失配 →
            #   思维链不着色）。think 态 r1 = 已确认的思考正文右端（分离器
            #   持回的闭标签候选尾部、流器空格启发式持回的尾词均不含），
            #   越界的 token 暂缓消费、待定性后再对齐；思考未闭合被截断/
            #   停止时由 finally 的 flush 兜底补齐（思维链仍完整着色）。
            if splitter.state == "probe":
                return
            r0 = splitter.reasoning_start()
            r1 = r0 + len(reasoning_cum)
            head = splitter.content_start()   # think 态 → None（正文未开始）
            fed = splitter.fed_len()          # 分离器已见文本上界（线程竞态护栏）
            # log_probs 滞后一步：仅对齐已有 ppl 的 token
            n_avail = min(len(streamer.token_ids), len(proc.log_probs))
            for i in range(seen, n_avail):
                prefix_text = self.tokenizer.decode(
                    streamer.token_ids[: i + 1], skip_special_tokens=True
                )
                piece = prefix_text[aligned_len:]
                end = aligned_len + len(piece)
                if end > fed:
                    # token 文本尚未经 chunk 喂入分离器（生成线程先于消费
                    # 推进）→ 暂缓定性，否则对齐会超前于 cum_text、契约 C3 失守
                    break
                if head is None and end > r1:
                    break  # 尾部尚未定性（可能是闭标签前缀）→ 暂缓消费
                aligned_len, seen = end, i + 1
                if not piece:
                    continue
                start = end - len(piece)
                p0, p1 = max(r0, start), min(r1, end)
                if p1 > p0:
                    r_aligned.append((piece[p0 - start: p1 - start], _ppl_of(i)))
                if head is not None and end > max(head, start):
                    aligned.append((piece[max(head, start) - start:], _ppl_of(i)))

        try:
            for chunk in streamer:
                if chunk:
                    r, c = splitter.feed(chunk)
                    reasoning_cum += r
                    cum_text += c
                if reasoning_cum:
                    self.reasoning_seen = True
                _pull_new_tokens()
                yield GenUpdate(
                    cum_text=cum_text,
                    reasoning_cum=reasoning_cum,
                    token_ppls=[p for _, p in aligned],
                    token_texts=[t for t, _ in aligned],
                    reasoning_token_ppls=[p for _, p in r_aligned],
                    reasoning_token_texts=[t for t, _ in r_aligned],
                    reasoning_closed=splitter.closed,
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
            # 流结束：未闭合的思考区尾部归位（probe→正文），再消费被扣住的 token
            r, c = splitter.flush()
            reasoning_cum += r
            cum_text += c
            _pull_new_tokens()
            _final_snapshot = GenUpdate(
                cum_text=cum_text,
                reasoning_cum=reasoning_cum,
                token_ppls=[p for _, p in aligned],
                token_texts=[t for t, _ in aligned],
                reasoning_token_ppls=[p for _, p in r_aligned],
                reasoning_token_texts=[t for t, _ in r_aligned],
                reasoning_closed=splitter.closed,
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


class LlamaCppBackend:
    """基于 llama-cpp-python 的本地 GGUF 量化模型后端。

    面向 llama.cpp 生态的各种量化 LLM 部署（Q4_K_M / Q5_K_M / Q8_0 / IQ 系列…）：
    - load：本地 .gguf 文件直接加载；非本地路径视为 HuggingFace GGUF 仓库 ID
      （Llama.from_pretrained 自动下载，如 Qwen/Qwen2.5-0.5B-Instruct-GGUF）。
    - 聊天模板：读取 GGUF 内嵌 tokenizer.chat_template，经 llama-cpp-python
      自带的 Jinja2ChatFormatter（jinja2）渲染，与 llama.cpp server 行为一致；
      模型未内嵌模板时报错并提示改用 prefix/raw 模式。
    - 流式生成：迭代 Llama.generate()（流式 token 生成器），每个 token 产出
      时经 llama_get_logits 读取产生该 token 的原始分布（惩罚/温度整形前，
      语义同 LocalBackend 的 ppl 采集）→ 逐 token 困惑度。不经
      create_completion(logprobs=..)——该路径要求 logits_all=True，会预分配
      n_ctx×vocab 的巨型 scores 数组（4k 上下文 × 15 万词表 ≈ 2.4GB），对
      量化小内存部署不友好。
    - 上下文困惑度：reset 后逐 token eval 前向打分 exp(mean CE)，与
      llama.cpp 官方 perplexity 工具同思路。
    - KV 缓存：Llama.generate 自带最长公共前缀复用（同一 Llama 对象连续
      生成），last_cache_info 汇报命中情况；prompt 超出 n_ctx 时保留
      尾部（最新上下文）。
    """

    kind = "llamacpp"

    def __init__(self):
        self._llm = None
        self.model_name = ""
        self.device = "cpu"
        self.quant_info = ""        # 量化信息（模型名 + Q4_K_M 等，元数据可得时）
        self.context_size = 0       # 实际生效的上下文窗口
        self.last_cache_info = ""   # 最近一次生成的缓存命中信息（UI 展示用）
        self._stop_event = threading.Event()
        self.reasoning_seen = False # 运行时检测：本模型曾输出思维链

    # ------------------------------------------------------- 思维链能力探测
    def reasoning_support(self) -> ReasoningSupport:
        """GGUF 元数据/模型名推断；unknown 留待运行时检测（simd 标签）。"""
        if not self.loaded:
            return {"supported": "unknown", "toggleable": False}
        if self.reasoning_seen:
            return {"supported": "yes", "toggleable": False}
        name = (self.model_name or "").lower()
        tpl = self._meta().get("tokenizer.chat_template", "") or ""
        if any(k in name for k in _REASONING_NAME_HINTS) or "enable_thinking" in tpl:
            return {"supported": "yes", "toggleable": "enable_thinking" in tpl}
        return {"supported": "unknown", "toggleable": False}

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        """加载 GGUF 模型。model_path 为本地 .gguf 文件或 HF GGUF 仓库 ID。

        n_gpu_layers：GPU offload 层数（-1=全部，0=纯 CPU）；
        n_ctx：上下文窗口（KV cache 容量）。
        """
        llama_cpp = _require_llama_cpp()
        path = (model_path or "").strip()
        if not path:
            raise ValueError(
                "GGUF 模型路径不能为空，例如 models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
                " 或 HF 仓库 Qwen/Qwen2.5-0.5B-Instruct-GGUF"
            )
        n_ctx = max(int(n_ctx), 512)
        kwargs = dict(n_ctx=n_ctx, n_gpu_layers=int(n_gpu_layers), verbose=False)
        if os.path.exists(path):
            self._llm = llama_cpp.Llama(model_path=path, **kwargs)
        elif "/" in path and path.lower().endswith(".gguf"):
            # 仓库 ID 带文件名：org/repo/file.gguf → 指定量化文件下载
            repo_id, _, filename = path.replace("\\", "/").rpartition("/")
            self._llm = llama_cpp.Llama.from_pretrained(
                repo_id=repo_id, filename=filename, **kwargs
            )
        else:
            # HuggingFace GGUF 仓库 ID（下载仓库内 *.gguf；仓库含多个量化文件
            # 时 from_pretrained 会报错并列出可用文件，改填 仓库ID/文件名.gguf）
            try:
                self._llm = llama_cpp.Llama.from_pretrained(
                    repo_id=path, filename="*.gguf", **kwargs
                )
            except Exception as e:  # noqa: BLE001
                if "Multiple files found" in str(e):
                    raise ValueError(
                        f"仓库 {path} 含多个 GGUF 量化文件，请在路径后追加文件名"
                        f"指定其一，例如 {path}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
                        f"\n（完整列表见原始报错）\n原始错误：{e}"
                    ) from e
                raise
        self.model_name = path
        self.context_size = self._n_ctx()
        gpu = int(n_gpu_layers)
        self.device = (
            "CPU" if gpu == 0
            else f"GPU×{gpu}层" if gpu > 0
            else "GPU(全部层)" if gpu == -1
            else "CPU"
        )
        self.quant_info = self._quant_desc()
        self._stop_event = threading.Event()
        self.last_cache_info = ""
        # llama-cpp-python 0.3.48 会把部分新架构（如 Qwen3.5）误判为 hybrid
        # 模型，默认启用 KV checkpoint：每次 prompt（len>1）都把整个 KV 快照
        # save_checkpoint（随上下文长度线性增长，300 token 时单次 ~0.5s），
        # 首 token 延迟因此 ~1s。本项目不做投机采样/rollback，checkpoint 无用，
        # 直接禁用（实测 1.09s → 0.09s）。Qwen3.5 是纯 transformer，无影响。
        try:
            _mgr = getattr(self._llm, "_hybrid_cache_mgr", None)
            if _mgr is not None:
                _mgr.max_checkpoints = 0
        except Exception:  # noqa: BLE001
            pass

    def unload(self):
        """释放 GGUF 模型与 KV 缓存，归还显存（切换后端时由服务端调用）。"""
        llm, self._llm = self._llm, None
        self.model_name = ""
        self.quant_info = ""
        self.context_size = 0
        self.last_cache_info = ""
        self.reasoning_seen = False
        if llm is not None:
            close = getattr(llm, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001  释放尽力而为，引用已断开
                    pass
            del llm  # 引用清零 → __del__ 释放 llama.cpp 上下文/权重

    # ------------------------------------------------------------ 底层访问
    def _n_ctx(self) -> int:
        """实际上下文窗口（新版 n_ctx 为方法，旧版为属性，兼容两者）。"""
        raw = getattr(self._llm, "n_ctx", 4096)
        try:
            return max(int(raw() if callable(raw) else raw), 512)
        except (TypeError, ValueError):
            return 4096

    def _n_vocab(self) -> int:
        """词表大小（新版 n_vocab 为方法，旧版为属性，兼容两者）。"""
        raw = getattr(self._llm, "n_vocab", 0)
        try:
            return int(raw() if callable(raw) else raw)
        except (TypeError, ValueError):
            return 0

    def _read_logits(self, np):
        """读取最近一次 decode 末位置的完整 logits（float64 副本，防覆盖）。

        不可用（接口不兼容）时返回 None → 困惑度退化为无数据。
        """
        llama_cpp = _require_llama_cpp()
        try:
            n_vocab = self._n_vocab()
            if n_vocab <= 0:
                return None
            ptr = llama_cpp.llama_get_logits(self._llm.ctx)
            if not ptr:
                return None
            return np.array(
                np.ctypeslib.as_array(ptr, shape=(n_vocab,)), dtype=np.float64
            )
        except Exception:  # noqa: BLE001
            return None

    def _is_eog(self, token) -> bool:
        """是否为结束生成 token（EOG，含 EOS）。"""
        try:
            llama_cpp = _require_llama_cpp()
            return bool(
                llama_cpp.llama_vocab_is_eog(self._llm._model.vocab, token)
            )
        except Exception:  # noqa: BLE001  老版本回退：仅比对 EOS id
            try:
                return token == self._llm.token_eos()
            except Exception:  # noqa: BLE001
                return False

    # ---------------------------------------------------------------- 元数据
    def _meta(self) -> dict:
        """GGUF 元数据（llama-cpp-python Llama.metadata，版本兼容防御）。"""
        try:
            return dict(getattr(self._llm, "metadata", None) or {})
        except Exception:  # noqa: BLE001
            return {}

    def _quant_desc(self) -> str:
        """模型名 + 量化类型（general.file_type → Q4_K_M 等）。"""
        m = self._meta()
        name = m.get("general.name") or ""
        q = ""
        raw_ft = m.get("general.file_type")
        if raw_ft is not None:
            try:
                code = int(str(raw_ft))
                names = {}
                for mod in (_llama_cpp, getattr(_llama_cpp, "llama_cpp", None)):
                    if mod is None:
                        continue
                    for attr in dir(mod):
                        if attr.startswith("LLAMA_FTYPE_MOSTLY_"):
                            names[getattr(mod, attr)] = attr[len("LLAMA_FTYPE_MOSTLY_"):]
                q = names.get(code, f"ftype={code}")
            except (TypeError, ValueError):
                q = ""
        return "｜".join(p for p in (name, q) if p)

    # -------------------------------------------------------------- 聊天模板
    def _detok_text(self, token_ids) -> str:
        try:
            return self._llm.detokenize(list(token_ids)).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    def _render_chat(self, messages, enable_thinking=None) -> str:
        """messages → GGUF 内嵌聊天模板渲染文本（含 assistant 起始标记）。

        enable_thinking 经 Jinja2ChatFormatter 的 **kwargs 注入模板上下文
        （llama-cpp-python 0.3.x 会透传给 Jinja render）。Qwen3 系模板据此
        决定是否预置未闭合的 <think>；模板无该变量时传参无副作用。
        """
        llama_cpp = _require_llama_cpp()
        template = self._meta().get("tokenizer.chat_template", "")
        if not template:
            raise ValueError(
                "该 GGUF 未内嵌 tokenizer.chat_template，chat 模式不可用；"
                "请改用 prefix/raw 上下文模式，或换用带聊天模板的 Instruct 量化模型"
            )
        lcf = llama_cpp.llama_chat_format
        formatter = lcf.Jinja2ChatFormatter(
            template=template,
            eos_token=self._detok_text([self._llm.token_eos()]),
            bos_token=self._detok_text([self._llm.token_bos()]),
        )
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            resp = formatter(messages=list(messages), **kwargs)  # 0.3.x：__call__
        except TypeError:
            # 旧版无 kwargs 透传 / 旧方法名：退化为不传开关
            call = getattr(formatter, "format_messages", None) or formatter
            try:
                resp = call(list(messages), **kwargs)
            except TypeError:
                resp = call(list(messages))
        return resp.prompt

    def apply_chat_template(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        """messages → 模板化生成前缀文本（含 assistant 起始标记）+ 活动块续写头。

        enable_thinking：透传给 GGUF 内嵌 Jinja 模板。以前此开关在 llama.cpp
        后端被丢弃，模板里 enable_thinking 未定义 → 走 else 分支渲染出空的
        <think></think>（等于显式关闭思考），导致「思考模式 on」也不出思维链。
        cot_prefix：被中断的思维链（正文尚未开始时的续写头），拼进思考区。
        """
        return (
            _splice_pending_think(self._render_chat(messages, enable_thinking), cot_prefix)
            + (active_text or "")
        )

    # ----------------------------------------------------------- prompt 构造
    def build_chat_prompt(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        """chat 模式：llama.cpp 后端 → 应用聊天模板后的平文本 str。"""
        return self.apply_chat_template(messages, active_text, enable_thinking, cot_prefix)

    def build_flat_prompt(self, flat_text):
        """prefix/raw 模式：llama.cpp 后端 → 原样平文本 str。"""
        return flat_text

    # ------------------------------------------------------------ token 工具
    def _tokenize(self, text: str) -> list:
        """文本 → token id 列表（special 解析，与库内字符串 prompt 路径一致）。

        tokenize 默认 add_bos=True；模板自身已内嵌 BOS 文本时会产生重复
        前导 BOS（llama-cpp-python 对此仅告警），此处显式去重。
        """
        ids = self._llm.tokenize(text.encode("utf-8"), special=True)
        try:
            bos = self._llm.token_bos()
            if bos is not None and len(ids) >= 2 and ids[0] == ids[1] == bos:
                ids = ids[1:]
        except Exception:  # noqa: BLE001
            pass
        return ids

    # -------------------------------------------------------------- 困惑度
    def compute_context_ppl(self, text: str) -> float:
        """上下文困惑度：分块批量前向打分 exp(mean CE)。提示词准确度指标。"""
        return self.score_context(text)[0]

    def score_context(self, context: str, tail_text: str = "", abort_event=None):
        """上下文打分（单趟 eval）：整段困惑度 + 尾部文本逐 token 困惑度。

        与 llama.cpp 官方 perplexity 工具同思路：reset 后分块 eval，块内
        逐位置从 llama_get_logits_ith 取完整分布，log_softmax 回填下一 token
        的 log-prob；超长文本保留尾部（最新上下文）。批量 eval（单次喂整块，
        GPU 矩阵乘满载）相比旧逐 token eval（每次单 token kernel，吞吐低
        数量级）大幅提速 prefill。打分后上下文即被评文本，同 prompt 的下一次
        生成可命中 KV 前缀缓存。内部 batch 接口不兼容时退化为逐 token eval
        （数值一致，仅慢）。

        tail_text（活动块手动编辑的文本，位于 context 末尾）在同一趟内取得
        其逐 token ppl，前端据此为编辑文本着色。返回契约同 LocalBackend。

        abort_event：可选停止信号。分块 eval 的块间/逐 token 迭代内检查，
        命中即中止打分（已 eval 的部分保持为 KV 前缀，下次生成可复用）→
        prefill 阶段停止也能即时生效，无需等整趟 eval 跑完。
        """
        if not self.loaded or not context.strip():
            return float("nan"), [], []
        if abort_event is not None and abort_event.is_set():
            return float("nan"), [], []
        try:
            import numpy as np
        except ImportError:
            return float("nan"), [], []
        toks = self._tokenize(context)
        if len(toks) < 2:
            return float("nan"), [], []
        limit = self._n_ctx() - 2
        if len(toks) > limit:
            toks = toks[-limit:]
        try:
            self._llm.reset()
            log_probs = self._ppl_batched(toks, np, abort_event=abort_event)
            if log_probs is None and (abort_event is None or not abort_event.is_set()):
                log_probs = self._ppl_one_by_one(toks, np, abort_event=abort_event)
        except Exception:  # noqa: BLE001  eval 接口不兼容等 → 退化为不可用
            return float("nan"), [], []
        if not log_probs:
            return float("nan"), [], []
        ctx = math.exp(min(-sum(log_probs) / len(log_probs), 20.0))
        return ctx, *self._score_tail(context, toks, log_probs, tail_text)

    def _score_tail(self, context, toks, log_probs, tail_text):
        """尾部文本逐 token 打分：log_probs[i-1] 为 toks[i] 的 log-prob。

        token 边界用「头部单独 tokenize 必须是全量 tokenize 的前缀」判定；
        文本片段用 detokenize 差分对齐（与 generate_stream 同语义）。
        超长截断（保留尾部）后头部不再是前缀 → 放弃（罕见）。
        """
        if not tail_text or not context.endswith(tail_text):
            return [], []
        head_toks = self._tokenize(context[: len(context) - len(tail_text)])
        start = len(head_toks)
        if start < 1 or start >= len(toks) or toks[:start] != head_toks:
            return [], []
        texts, ppls, full = [], [], ""
        for i in range(start, len(toks)):
            new = self._llm.detokenize(
                toks[start: i + 1], prev_tokens=toks[:start]
            ).decode("utf-8", errors="ignore")
            piece = new[len(full):]
            full = new
            if not piece:
                continue
            texts.append(piece)
            # log_probs[j] 对应 toks[j+1]；i ≤ len(toks)-1 < len(log_probs)+1
            ppls.append(math.exp(min(-log_probs[i - 1], 20.0)))
        if not texts or not tail_text.endswith("".join(texts)):
            return [], []  # 对齐失配（尾部含特殊 token 等）→ 放弃
        return texts, ppls

    def _ppl_one_by_one(self, toks, np, abort_event=None):
        """逐 token eval 打分（兼容回退路径）。"""
        log_probs = []
        for i in range(len(toks) - 1):
            if abort_event is not None and abort_event.is_set():
                return []  # 停止：中止打分
            self._llm.eval([toks[i]])
            logits = self._read_logits(np)
            if logits is None:
                return []
            m = float(logits.max())
            lse = m + math.log(float(np.exp(logits - m).sum()))
            log_probs.append(float(logits[toks[i + 1]]) - lse)
        return log_probs

    def _ppl_batched(self, toks, np, abort_event=None):
        """分块批量 eval 打分（llama-cpp-python 内部 batch 接口）。

        逐位置 logits 需在 batch 上显式置位（公共 eval() 只给末位产出
        logits），故镜像 Llama.eval 的内部流程：_batch.add_sequence 全置
        logits → _decode_eval_batch → llama_get_logits_ith 逐位置读取，
        并同步 input_ids/n_tokens 台账保证后续 generate 前缀复用。
        接口不兼容（版本差异）返回 None → 调用方回退逐 token 路径。

        abort_event：可选停止信号。块间检查，命中返回 None（已 eval 的
        前缀保持为 KV 缓存，下次生成可复用）。
        """
        llama_cpp = _require_llama_cpp()
        get_ith = getattr(llama_cpp, "llama_get_logits_ith", None)
        llm = self._llm
        if get_ith is None or not all(
            hasattr(llm, a) for a in ("_batch", "_ctx", "input_ids", "n_tokens")
        ):
            return None
        try:
            decode = llm._decode_eval_batch
        except AttributeError:
            return None
        n_vocab = self._n_vocab()
        if n_vocab <= 0:
            return None
        # 块长上限 256：C 侧 logits 输出缓冲 = 块长×词表（Qwen 词表 15 万 ×
        # 2048 块 ≈ 1.2GB，256 ≈ 150MB，够快也够省）
        bs = max(1, min(256, int(getattr(llm, "n_batch", 256) or 256)))
        log_probs = []
        end = len(toks) - 1  # 末 token 无"下一 token"可打分，无需 eval
        pos = 0
        while pos < end:
            if abort_event is not None and abort_event.is_set():
                return None  # 停止：中止打分，已 eval 前缀保持为 KV 缓存
            chunk = toks[pos: min(pos + bs, end)]
            n = len(chunk)
            llm._batch.reset()
            llm._batch.add_sequence(
                chunk,
                list(range(llm.n_tokens, llm.n_tokens + n)),
                [0],
                [True] * n,
            )
            for cleanup in ("clear_loras", "clear_cvec"):
                fn = getattr(llm._ctx, cleanup, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
            got = int(decode(chunk, n))
            if got <= 0:
                return None
            for j in range(got):
                ptr = get_ith(llm.ctx, j)
                if not ptr:
                    return None  # 版本行为不符（未置位的 logits）→ 整体退化
                logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
                nxt = toks[pos + j + 1]
                m = float(logits.max())
                lse = m + math.log(float(np.exp(logits - m).sum()))
                log_probs.append(float(logits[nxt]) - lse)
            # 与 Llama.eval 相同的台账同步（generate 前缀复用依赖）
            llm.input_ids[pos: pos + got] = chunk[:got]
            llm.n_tokens += got
            pos += got
        return log_probs

    # ---------------------------------------------------------------- 生成
    def stop(self):
        """请求优雅停止：流式循环在每个 token 间检查事件，自然收尾。"""
        self._stop_event.set()

    def _cache_note(self, prompt_tokens) -> str:
        """生成前的 KV 前缀缓存命中预估（Llama.generate 按最长公共前缀复用）。"""
        try:
            ids = getattr(self._llm, "_input_ids", None)
            if ids is None:
                ids = getattr(self._llm, "input_ids", None)
            n = int(getattr(self._llm, "n_tokens", 0) or 0)
            if ids is None or n <= 0:
                return "KV缓存未命中"
            ctx_ids = [int(t) for t in list(ids)[:n]]
        except Exception:  # noqa: BLE001
            return ""
        k = _common_prefix_len(ctx_ids, prompt_tokens)
        if k < _MIN_CACHE_REUSE:
            return "KV缓存未命中"
        return f"⚡KV缓存复用 {k}/{len(prompt_tokens)} token"

    def generate_stream(self, context: str, **gen_params):
        """流式生成。yield GenUpdate；自然结束/停止后额外 yield final=True 快照。

        迭代 Llama.generate()：token 逐个产出（生成惰性推进，stop 检查即时
        生效），每个 token 产出时 llama_get_logits 恰为产生该 token 的原始
        分布（惩罚/温度整形前）→ log_softmax 取该 token 的 log-prob 即逐
        token 困惑度；接口不可用时自动退化为无困惑度。
        gen_params: max_new_tokens / do_sample / temperature / top_k / top_p /
        repetition_penalty（映射为 llama.cpp 的 repeat_penalty）。
        """
        if not self.loaded:
            raise RuntimeError("模型尚未加载，请先在顶栏加载 GGUF 模型")

        import numpy as np

        self._stop_event = threading.Event()
        do_sample = bool(gen_params.get("do_sample", True))
        max_tokens = int(gen_params.get("max_new_tokens", 256))

        # 上下文窗口管理：超长 prompt 保留尾部（最新上下文），max_tokens 不越界
        toks = self._tokenize(context)
        n_ctx = self._n_ctx()
        if len(toks) > n_ctx - 8:
            toks = toks[-(n_ctx - 8):]
        max_tokens = max(1, min(max_tokens, n_ctx - len(toks) - 1))
        self.last_cache_info = self._cache_note(toks)

        # llama-cpp-python 0.3.x 的 generate() 在前缀完全命中缓存长度
        # （longest_token_prefix == n_tokens）时不会缩短 tokens：截断分支的条件是
        # `longest_prefix < n_tokens`，等于时直接跳过 → 整个 prompt 被重新 eval
        # 一遍，回灌长思维链时 prefill 明显变慢（实测 300 token 思维链首 token
        # ~1.07s，尽管 KV 明明全部命中）。这里手动处理：完全命中时把 tokens
        # 缩短到新增部分，并以 reset=False 续跑——n_tokens 保留，eval 从缓存
        # 末尾继续，只 prefill 新增的闭标签等少量 token。
        reuse_prefix = 0
        llm = self._llm
        if (
            hasattr(llm, "longest_token_prefix")
            and getattr(llm, "n_tokens", 0) > 0
            and len(toks) > int(llm.n_tokens)
        ):
            try:
                if llm.longest_token_prefix(llm._input_ids, toks) == int(llm.n_tokens):
                    reuse_prefix = int(llm.n_tokens)
            except Exception:  # noqa: BLE001
                reuse_prefix = 0
        if reuse_prefix:
            toks = toks[reuse_prefix:]

        gen_kwargs = dict(
            top_k=int(gen_params.get("top_k", 50)),
            top_p=float(gen_params.get("top_p", 0.95)),
            temp=float(gen_params.get("temperature", 0.8)) if do_sample else 0.0,
            repeat_penalty=float(gen_params.get("repetition_penalty", 1.1)),
        )
        if reuse_prefix:
            gen_kwargs["reset"] = False
        try:
            gen = self._llm.generate(toks, **gen_kwargs)
        except TypeError:
            # 老实现/替身不接受 reset 参数：退化为全量 prefill（正确性不变）
            gen_kwargs.pop("reset", None)
            gen = self._llm.generate(toks, **gen_kwargs)

        cum_text = ""        # 仅正文（思维链经 _ThinkSplitter 分离）
        reasoning_cum = ""
        full_text = ""       # 含思维链的完整解码文本（差分基准）
        token_texts: list = []
        token_ppls: list = []
        reasoning_token_texts: list = []
        reasoning_token_ppls: list = []
        completion_tokens: list = []
        # 模板已预置 <think> 时流首即思考正文，分离器需直接进入 think 态
        splitter = _ThinkSplitter(assume_thinking=_prompt_pending_think(context))
        last_r_ppl = None   # 最近一个思考区 token 的 ppl（flush 补尾时沿用）
        n_gen = 0
        try:
            for token in gen:
                if self._stop_event.is_set():
                    break
                if self._is_eog(token):  # 结束生成，不计入文本/困惑度
                    break
                # 困惑度：此刻 logits 缓冲区恰为产生该 token 的原始分布
                logits = self._read_logits(np)
                ppl = None
                if logits is not None:
                    m = float(logits.max())
                    lse = m + math.log(float(np.exp(logits - m).sum()))
                    lp = float(logits[token]) - lse
                    ppl = math.exp(min(-lp, 20.0))
                # token-文本对齐（增量解码差分法，同 LocalBackend）：逐 token
                # 解码前缀取差分，中文多字节字符跨 token 不乱码（未完成字节
                # 暂缺、完成后经差分补齐，"".join(token_texts)==cum_text 恒成立）。
                # 思考区 token 同样算 ppl（思维链也参与困惑度着色）
                completion_tokens.append(token)
                new_text = self._llm.detokenize(
                    completion_tokens, prev_tokens=toks
                ).decode("utf-8", errors="ignore")
                piece = new_text[len(full_text):]
                full_text = new_text
                r, c = splitter.feed(piece)
                reasoning_cum += r
                cum_text += c
                if r:
                    self.reasoning_seen = True
                    reasoning_token_texts.append(r)
                    reasoning_token_ppls.append(ppl)
                    last_r_ppl = ppl
                if c:
                    token_texts.append(c)
                    token_ppls.append(ppl)
                yield GenUpdate(
                    cum_text=cum_text,
                    reasoning_cum=reasoning_cum,
                    token_ppls=list(token_ppls),
                    token_texts=list(token_texts),
                    reasoning_token_ppls=list(reasoning_token_ppls),
                    reasoning_token_texts=list(reasoning_token_texts),
                    reasoning_closed=splitter.closed,
                )
                n_gen += 1
                if n_gen >= max_tokens:
                    break
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            # 流结束：未闭合的思考区尾部归位
            r, c = splitter.flush()
            reasoning_cum += r
            cum_text += c
            # flush 补回的是 think 态暂缓的闭标签候选尾部（最长 len(闭标签)-1）。
            # 必须同步补进 token 序列：否则 join(reasoning_token_texts) 比
            # reasoning_cum 短这几个字符，插件端对账失配 → 思维链不着色，
            # 定稿冻结校验（join == cot 块内容）也会永远失败。
            if r:
                reasoning_token_texts.append(r)
                reasoning_token_ppls.append(last_r_ppl)
        yield GenUpdate(
            cum_text=cum_text, reasoning_cum=reasoning_cum,
            token_ppls=token_ppls, token_texts=token_texts,
            reasoning_token_ppls=reasoning_token_ppls,
            reasoning_token_texts=reasoning_token_texts,
            reasoning_closed=splitter.closed, final=True,
        )


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
        self.supports_logprobs = None
        self.reasoning_seen = False  # 运行时检测：响应含 reasoning_content 即支持
        self._stop_event = threading.Event()
        self._resp = None

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return bool(self.model and self.base_url)

    def unload(self):
        """断开 API 会话（无本地资源，仅清空连接配置）。"""
        self.stop()
        self.base_url = ""
        self.api_key = ""
        self.model = ""
        self.supports_logprobs = None
        self.reasoning_seen = False

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
        self.reasoning_seen = False

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
    def build_chat_prompt(self, messages, active_text, enable_thinking=None, cot_prefix=""):
        """chat 模式：API 后端 → messages 列表。活动块文本作为末尾 assistant
        消息（prefill），模型以其为上文继续输出新内容。
        enable_thinking 由 generate_stream 落到请求参数（vLLM/SGLang 的
        chat_template_kwargs），此处仅统一三后端签名。
        cot_prefix：活动单元的思维链 → 走 OpenAI 扩展字段 reasoning_content
        （DeepSeek / vLLM / SGLang 约定），正文续写时一并在 assistant 消息
        携带，保证续写上下文与首次生成一致。个别服务端不认该字段时，效果
        退化为"不带思维链上文"，即与修复前一致，不会报错。"""
        msgs = [dict(m) for m in messages]
        cot = (cot_prefix or "").strip()
        if (active_text or "").strip():
            a = {"role": "assistant", "content": active_text}
            if cot:
                a["reasoning_content"] = cot
            msgs.append(a)
        elif cot:
            msgs.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": cot,
            })
        return msgs

    def build_flat_prompt(self, flat_text):
        """prefix/raw 模式：平文本包成单条 user 消息。"""
        return [{"role": "user", "content": flat_text}]

    # ------------------------------------------------------- 思维链能力探测
    def reasoning_support(self) -> ReasoningSupport:
        """模型名启发式；unknown 留待运行时检测（响应含 reasoning_content）。"""
        if not self.loaded:
            return {"supported": "unknown", "toggleable": False}
        if self.reasoning_seen:
            return {"supported": "yes", "toggleable": True}
        name = (self.model or "").lower()
        if any(k in name for k in _REASONING_NAME_HINTS):
            return {"supported": "yes", "toggleable": True}
        return {"supported": "unknown", "toggleable": True}

    # -------------------------------------------------------------- 困惑度
    def compute_context_ppl(self, prompt) -> float:
        """API 无法获取 prompt token 概率，上下文困惑度不可用。"""
        return float("nan")

    # ---------------------------------------------------------------- 生成
    def stop(self):
        self._stop_event.set()
        if self._resp is None:
            return
        # 打断读线程的阻塞 recv：Windows 上 shutdown() 唤不醒另一线程的 recv，
        # 而 close() 更会让调用方卡死（读线程正阻塞在 recv 时，实测会挂住，
        # 见 tests/dbg_shutdown.py）。改为把底层 socket 读超时设为短值——
        # SO_RCVTIMEO 是 fd 级选项，阻塞中的 recv 会在超时后抛异常，读线程
        # 捕获后视作正常停止，并在其 finally 中关闭连接，资源即刻回收；
        # 主生成器则在 ≤0.2s 的队列轮询周期内收到停止信号收尾。
        try:
            conn = getattr(getattr(self._resp, "raw", None), "_connection", None)
            sock = getattr(conn, "sock", None)
            if sock is not None:
                sock.settimeout(0.5)
        except Exception:  # noqa: BLE001
            pass

    def _post_stream(self, payload):
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url, headers=self._headers(), json=payload, stream=True, timeout=(15, 600)
        )
        # 400 降级重试：logprobs / chat_template_kwargs 服务端不支持时去掉再试
        if resp.status_code == 400 and (
            "logprobs" in payload or "chat_template_kwargs" in payload
        ):
            dropped = [k for k in ("logprobs", "chat_template_kwargs") if k in payload]
            resp.close()
            if "logprobs" in dropped:
                self.supports_logprobs = False
            for k in dropped:
                payload.pop(k)
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
        """流式生成。prompt 为 messages 列表；yield GenUpdate。

        实现为「读线程 + 队列」：请求响应由独立读线程迭代解析并写入队列，
        主生成器以 0.2s 超时轮询队列，每次超时检查停止信号。这样即使服务端
        长时间不产 token（静默期，如思考/缓冲），/api/stop 也能在下一个轮询
        周期内生效——主线程不再被阻塞 recv 卡死（Windows 上关闭 fd 无法打断
        另一线程的阻塞读，见 tests/dbg_shutdown.py 与 repro_stop_api_silent.py）。
        停止时读线程可能仍阻塞在 socket 上：stop() 通过把 socket 读超时设为
        短值（SO_RCVTIMEO）令其超时退出，读线程 finally 关闭连接并回收资源。
        """
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
        # 思考模式关闭（vLLM/SGLang 约定的 chat_template_kwargs；OpenAI
        # 官方等不认识该参数 → 400 时自动降级去掉重试）
        if gen_params.get("enable_thinking") is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        resp = self._post_stream(payload)
        self._resp = resp
        if "logprobs" in payload and self.supports_logprobs is None:
            self.supports_logprobs = True  # 200 通过，待首条数据确认

        items: "queue.Queue" = queue.Queue()
        _DONE = object()  # 哨兵：流正常结束 / 读线程异常

        def _reader():
            """读线程：迭代 SSE 行 → 解析 → 入队（仅队列与停止信号交互）。"""
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
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    # 思维链（DeepSeek/GLM/Qwen 的 reasoning_content；部分服务
                    # 用 reasoning 字段）。API 不返回 reasoning 的 logprobs，
                    # 思维链困惑度着色仅本地/llama.cpp 后端可用
                    rpiece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if rpiece:
                        self.reasoning_seen = True
                    # logprobs 逐 token 困惑度（服务端支持时）
                    lps = []
                    for ent in (choice.get("logprobs") or {}).get("content") or []:
                        tok = ent.get("token") or ""
                        logprob = ent.get("logprob")
                        if tok and logprob is not None:
                            lps.append((tok, math.exp(min(-logprob, 20.0))))
                    if piece or rpiece or lps:
                        items.put((piece, rpiece, lps))
            except (ValueError, requests.RequestException):
                # stop() 关闭连接导致迭代异常 → 视为正常停止，入队哨兵即可；
                # 真实网络错误 → 入队异常对象，主生成器重新抛出（与旧行为一致）
                if not self._stop_event.is_set():
                    items.put(sys.exc_info()[1])
            finally:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                items.put(_DONE)

        threading.Thread(target=_reader, daemon=True, name="api-stream-reader").start()

        cum_text = ""
        reasoning_cum = ""
        token_texts: list = []
        token_ppls: list = []
        final_snapshot = None
        try:
            while True:
                try:
                    item = items.get(timeout=0.2)
                except queue.Empty:
                    # 队列空且停止信号未置位 → 继续等（静默期也保持响应）
                    if self._stop_event.is_set():
                        break
                    continue
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item  # 读线程遇到的真实网络错误
                piece, rpiece, lps = item
                if piece:
                    cum_text += piece
                if rpiece:
                    reasoning_cum += rpiece
                for tok, ppl in lps:
                    token_texts.append(tok)
                    token_ppls.append(ppl)
                yield GenUpdate(
                    cum_text=cum_text,
                    reasoning_cum=reasoning_cum,
                    token_ppls=list(token_ppls),
                    token_texts=list(token_texts),
                    # API 服务端的 reasoning_content 是协议级字段，天然闭合
                    reasoning_closed=bool(reasoning_cum),
                )
        finally:
            self._resp = None
            if self.supports_logprobs and not token_ppls:
                self.supports_logprobs = False  # 200 但从未返回 logprobs 数据
            final_snapshot = GenUpdate(
                cum_text=cum_text, reasoning_cum=reasoning_cum,
                token_ppls=token_ppls, token_texts=token_texts,
                reasoning_closed=bool(reasoning_cum), final=True,
            )
        yield final_snapshot

# 🖋 生成式文本编辑器

基于 LLM 的分块式生成文本编辑器：提示词块引导生成，生成块流式续写、可随时暂停与手动编辑。类 Jupyter 的交互体验，专为"人机协作写作"设计。

## 功能特性

### 分块文档编辑
- **提示词块**（`<prompt>`）：用户编辑，指引 LLM 生成方向
- **生成块**（`<generate>`）：LLM 流式输出，也随时可暂停后手动编辑再续写
- 生成仅在最后一块生成块（活动生成单元）中进行，一键定稿开启新块
- **块锁定**：任意块可锁定，锁定后折叠为可展开标题栏且不可编辑；新增生成块时自动锁定前序全部块，聚焦当前写作
- **定时自动保存**：每 60 秒自动落盘（仅内容变化时），`saves/` 目录轮转仅保留最近 20 份，可一键关闭
- 保存/加载为带 XML 标记的 Markdown：

  ```markdown
  <prompt>
  写一段关于秋天的散文开头，100字左右。
  </prompt>

  <generate>
  秋日的午后，阳光穿过梧桐叶洒在青石板上……
  </generate>
  ```

### 双后端
- **本地后端**（transformers）：支持任意 HuggingFace / 本地路径模型，CUDA fp16 / CPU fp32 自适应
- **API 后端**（OpenAI 兼容）：覆盖 OpenAI / DeepSeek / Qwen / GLM / Kimi，以及 vLLM / Ollama / llama.cpp server 等本地推理服务，SSE 流式

### 困惑度（Perplexity）分析
- **上下文困惑度**：一次前向传播评估提示词与模型的匹配度（仅本地模式）
- **逐 token 生成困惑度**：本地用 LogitsProcessor 采集；API 模式经 logprobs 自动探测（不支持时降级）
- **伪彩色热力图**：绿（模型确定）→ 红（模型困惑），覆盖**全部生成块**
- 折线图 + 滑动平均 + 全文档几何平均困惑度
- **编辑容错**：手动编辑后，仅编辑点之后的着色失效（灰显），编辑点之前的保留；新生成部分继续着色

### KV 前缀缓存（本地后端）
- 保存上次生成的 KV Cache，续写/再次生成时按最长公共前缀复用，仅对新增 token 增量 prefill
- 长文本场景首 token 延迟实测约 **1/3**（2.8x 加速），贪心输出与全量 prefill 完全一致
- 生成线程串行化保护，停止后立即重启生成不会产生缓存竞态
- 状态栏实时显示「⚡KV缓存复用 X/Y token」

### 上下文模式
| 模式 | 说明 | 适用 |
|---|---|---|
| `chat` | 提示词块→user、生成块→assistant，经聊天模板包装 | Instruct 模型（默认） |
| `prefix` | 各块加【指令】/【正文】前缀后拼接 | base 模型 |
| `raw` | 全部块原文裸拼接 | 纯续写场景 |

### Agent Skill 接入
- 支持 Anthropic 风格 `SKILL.md`（YAML frontmatter + markdown 正文）与普通 `.md` 提示词文件
- 放入 `./skills/` 目录或直接上传，勾选后作为系统指令注入生成上下文

## 安装与运行

```bash
pip install -r requirements.txt

# HuggingFace 无法直连时可使用镜像
set HF_ENDPOINT=https://hf-mirror.com

python app.py
# 浏览器打开 http://127.0.0.1:7860
```

在顶栏选择后端模式：
- **本地模型**：填写模型路径或 HF ID（默认 `Qwen/Qwen2.5-0.5B-Instruct`），点击「加载模型」
- **API**：填写 base_url（到 `/v1` 层级）、API Key、模型名，点击「连接 API」

## 项目结构

```
├── app.py            # Gradio 前端：分块编辑、困惑度可视化、技能库面板
├── backend.py        # LLM 后端：LocalBackend（KV缓存）/ OpenAICompatBackend（SSE）
├── skills.py         # 技能扫描、解析（SKILL.md frontmatter）、上下文拼接
├── skills/           # 技能库目录
│   └── 中文散文写作/SKILL.md
├── saves/            # 文档保存目录（运行时生成）
├── requirements.txt
├── kv_cache_test.py      # KV 缓存回归：命中加速/贪心一致性/停止后再生成
├── ppl_accum_test.py     # 困惑度累积回归：跨块/编辑容错/序列化兼容
├── lock_autosave_test.py # 块锁定 + 定时自动保存逻辑回归
└── regress3_test.py      # 服务器 REST 回归（需 app.py 已启动）
```

## 已知约束

- Windows IME 兼容：块编辑用 `blur` 事件提交（`change` 会在输入法确认候选词时触发组件重建）
- transformers 建议 4.44–4.49（5.x 与 torch 2.5 存在 `CPUOffloadPolicy` 兼容问题）
- API 模式下上下文困惑度不可用（Chat Completions 不返回 prompt token 概率）；DeepSeek 等不支持 logprobs 的服务自动降级为无逐 token 困惑度

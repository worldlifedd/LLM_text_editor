# 🖋 生成式文本编辑器

基于 LLM 的分块式生成文本编辑器：提示词块引导生成，生成块流式续写、可随时暂停与手动编辑。类 Jupyter 的交互体验，专为"人机协作写作"设计。

提供两种前端，共享同一套后端与文档模型：
- **Gradio 网页前端**（`app.py`）：浏览器内编辑，带全量困惑度可视化
- **VSCode 插件前端**（`vscode-extension/`）：在纯 Markdown 里写作（提示词为 `<!-- prompt -->` 注释，生成文本为可见正文），习惯 VSCode 编辑的用户首选

## 功能特性

### 分块文档编辑
- **提示词块**（`<!-- prompt -->` 注释）：用户编辑，指引 LLM 生成方向
- **生成块**（可见正文）：LLM 流式输出，也随时可暂停后手动编辑再续写
- 生成仅在最后一块生成块（活动生成单元）中进行，一键定稿开启新块
- **块锁定**：任意块可锁定，锁定后折叠为可展开标题栏且不可编辑；新增生成块时自动锁定前序全部块，聚焦当前写作
- **定时自动保存**：每 60 秒自动落盘（仅内容变化时），`saves/` 目录轮转仅保留最近 20 份，可一键关闭
- 保存/加载为纯 Markdown（提示词即注释，生成内容即正文，渲染完全干净）：

  ```markdown
  <!-- prompt
  写一段关于秋天的散文开头，100字左右。
  -->

  秋日的午后，阳光穿过梧桐叶洒在青石板上……
  ```

### 三后端
- **本地后端**（transformers）：支持任意 HuggingFace / 本地路径模型，CUDA fp16 / CPU fp32 自适应
- **llama.cpp 后端**（`llama-cpp-python`）：进程内加载 GGUF 量化模型（Q4_K_M / Q5_K_M / Q8_0 / IQ 系列等），小显存 / 纯 CPU 也能跑大模型；支持 GPU offload 层数（`n_gpu_layers`）与上下文窗口（`n_ctx`）调节；聊天模板读取 GGUF 内嵌 `tokenizer.chat_template` 渲染（与 llama.cpp server 行为一致）；逐 token / 上下文困惑度可用；KV 前缀缓存由 `Llama.generate` 自动复用
- **API 后端**（OpenAI 兼容）：覆盖 OpenAI / DeepSeek / Qwen / GLM / Kimi，以及 vLLM / Ollama / llama.cpp server 等本地推理服务，SSE 流式

### 困惑度（Perplexity）分析
- **上下文困惑度**：一次前向传播评估提示词与模型的匹配度（本地 / llama.cpp 模式）
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
```

### 方式一：Gradio 网页前端

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

在顶栏选择后端模式：
- **本地模型**：填写模型路径或 HF ID（默认 `Qwen/Qwen2.5-0.5B-Instruct`），点击「加载模型」
- **llama.cpp（GGUF 量化模型）**：填写本地 `.gguf` 文件路径（如 `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`）或 HF GGUF 仓库 ID（如 `Qwen/Qwen2.5-0.5B-Instruct-GGUF`，首次自动下载），调节 `n_gpu_layers` / `n_ctx` 后点击「加载 GGUF」；需 `pip install llama-cpp-python`
- **API**：填写 base_url（到 `/v1` 层级）、API Key、模型名，点击「连接 API」

### 方式二：VSCode 插件前端

插件**自带 Python 无头服务端**（打包在扩展目录 `python/` 内），安装后无需手动启动、无需指定 `server.py` 路径——首次打开面板时插件会自动拉起服务。

1. 安装插件：在 VSCode 中「扩展」→「…」→「从 VSIX 安装」
   （源码调试则在 `vscode-extension/` 内 `npm install` 后按 F5；
   从源码打包 VSIX：`npx vsce package`，`python/` 服务端会自动从仓库根同步）

2. 确保本机 Python 已安装服务依赖（只需一次；跑本地模型还需 torch/transformers，跑 GGUF 量化模型还需 llama-cpp-python）：

   ```bash
   pip install fastapi uvicorn pyyaml requests
   pip install llama-cpp-python   # llama.cpp 模式（GGUF 量化模型）
   ```

3. 在侧边栏 **GTE 生成控制面板** 中连接服务、加载模型 / 连接 API、勾选技能并调整生成参数

> 若使用仓库根目录的 `server.py`（例如调试最新改动），可在设置 `gte.serverScript` 中指定其绝对路径，插件将优先使用。

#### 文档格式

插件以纯 Markdown 为文档载体：提示词块是 `<!-- prompt ... -->` 注释（渲染不可见），生成块是可见正文，每块生成正文前有一行 `<!-- generate -->` 注释作为块边界标记（用于分隔相邻生成块、标记空的活动生成单元）：

```markdown
<!-- prompt
写一段关于秋天的散文开头，100字左右。
-->

秋日的午后，阳光穿过梧桐叶洒在青石板上……

<!-- generate -->

（定稿后新开的活动生成块，生成从这里继续）
```

提示词注释也支持单行写法 `<!-- prompt: 指令内容 -->`。生成仅发生在**最后一块生成块**（活动生成单元）中：文档末尾没有生成块时按 `Ctrl+Enter` 会自动创建。

**系统提示词块**（`<!-- system ... -->` 注释）：技能可经命令 `GTE: 固化技能为系统提示词块`（Gradio 端为「📥 固化选中技能为系统块」按钮）固化为文档顶部的系统块，生成时作为系统级指令拼入上下文。固化后文档自包含——换到没有该技能的环境（另一台机器 / 另一个 skills 目录）也能完整复现生成过程：

```markdown
<!-- system
# 技能指令: 中文散文写作
# 说明: 讲究意象与节奏的中文散文写作技能
……技能完整指令体……
-->

<!-- prompt
写一段关于秋天的散文开头，100字左右。
-->

秋日的午后，阳光穿过梧桐叶洒在青石板上……
```

**思维链块**（`<!-- cot ... -->` 注释）：推理模型（DeepSeek-R1 / GLM / Qwen3 等）生成时，思考过程与正文分离——思维链流式写入活动生成块前紧邻的 cot 注释块（Markdown 渲染中不可见，源码中可查看/编辑），正文照常进入生成块。三后端均支持：API 模式捕获 `reasoning_content` 流；本地/llama.cpp 模式解析内嵌 `simd…skill` 标签。上下文组装**不包含** cot 块（推理模型会自行重新思考，回灌旧思维链反而干扰）：

```markdown
<!-- prompt
写一段关于秋天的散文开头，100字左右。
-->

<!-- cot
（模型思考过程，随文档保存、渲染不可见）
-->

秋日的午后，阳光穿过梧桐叶洒在青石板上……
```

Gradio 端在生成块上方提供「🧠 思维链（当前轮）」流式单元格，定稿时随块保存为 cot 块；历史思维链块默认折叠，展开可编辑。

**思维链能力探测与思考模式开关**：模型加载后前端显示思维链能力（✅ 支持 / ❓ 待检测，`❓` 在首次生成检测到思维链后自动转为 ✅；本地模式依据聊天模板是否含 `enable_thinking` 变量、模型名是否命中推理系关键词推断）。生成参数提供「🧠 思考模式」三态开关（auto/on/off）：本地模式经聊天模板 `enable_thinking` 变量生效（Qwen3 系），API 模式发送 `chat_template_kwargs`（vLLM/SGLang 约定，服务端不认识时自动降级），llama.cpp 由 GGUF 模板决定。

**思维链困惑度**：本地/llama.cpp 后端思考区 token 的 log-prob 与正文同源，思维链块同样按绿→红色阶着色（VSCode 插件与状态栏指标；定稿后随块冻结）；API 后端不返回 reasoning 的 logprobs，思维链不着色。

#### 命令与快捷键

| 命令 | 快捷键 | 说明 |
|---|---|---|
| `GTE: 生成 / 停止（同一按键切换）` | `Ctrl+Enter` | 在活动生成块中流式续写；生成中再按即优雅停止（状态栏持续显示 LLM 工作状态） |
| `GTE: 停止生成` | `Ctrl+Alt+Enter` | 停止当前生成，保留已生成文本 |
| `GTE: 定稿当前块并开启新块` | `Ctrl+Shift+Enter` | 锁定前序块、开启新的活动生成块 |
| `GTE: 追加提示词块` | `Ctrl+Alt+P` | 在文档末尾追加 `<!-- prompt -->` 注释块 |
| `GTE: 固化技能为系统提示词块` | — | 选择技能，固化插入为文档顶部 `<!-- system -->` 注释块（文档自包含、可移植复现） |
| `GTE: 锁定/解锁光标所在块` | — | 折叠该块，防止误编辑 |
| `GTE: 新建生成式文本文档` | — | 打开带初始模板的新 Markdown 文档 |
| `GTE: 打开生成控制面板` | — | 聚焦侧边栏控制面板 |

流式生成的 token 按困惑度着色（绿=确定、红=困惑），手动编辑后颜色自动灰显回退。

#### 插件配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `gte.serverUrl` | `http://127.0.0.1:8907` | 无头服务地址 |
| `gte.autoStartServer` | `true` | 服务不可达时自动用 python 拉起 server.py |
| `gte.pythonCommand` | `python` | 启动 server.py 所用 Python 命令 |
| `gte.serverScript` | 扩展内 `python/` | server.py 绝对路径（留空用随包自带服务端） |
| `gte.params` | 见面板 | 生成参数默认值 |

## 项目结构

```
├── app.py            # Gradio 前端：分块编辑、困惑度可视化、技能库面板
├── server.py         # FastAPI 无头服务：REST + SSE，供 VSCode 插件等前端使用
├── core.py           # 共享纯逻辑：文档序列化、上下文组装、困惑度聚合
├── backend.py        # LLM 后端：LocalBackend（KV缓存）/ LlamaCppBackend（GGUF 量化）/ OpenAICompatBackend（SSE）
├── skills.py         # 技能扫描、解析（SKILL.md frontmatter）、上下文拼接
├── skills/           # 技能库目录
│   └── 中文散文写作/SKILL.md
├── saves/            # 文档保存目录（运行时生成）
├── vscode-extension/ # VSCode 插件前端
│   ├── src/          # 插件源码（TS）：extension/generation/docmodel/panel/… 
│   ├── python/       # 构建产物：打包时从仓库根自动复制（不入 git）
│   ├── package.json  # 命令、快捷键、配置项声明
│   └── esbuild.js    # 构建/打包脚本（编译 + 同步 python/）
├── requirements.txt
├── kv_cache_test.py      # KV 缓存回归：命中加速/贪心一致性/停止后再生成
├── ppl_accum_test.py     # 困惑度累积回归：跨块/编辑容错/序列化兼容
├── lock_autosave_test.py # 块锁定 + 定时自动保存逻辑回归
├── regress3_test.py      # 服务器 REST 回归（需 app.py 已启动）
├── server_test.py        # 无头服务 REST/SSE 回归（需 server.py 已启动）
├── test_llamacpp_mock.py # LlamaCppBackend mock 单测（无需模型/llama-cpp-python）
└── test_llamacpp_model.py # llama.cpp 真实模型回归（GGUF_MODEL 指定模型，缺失时跳过）
```

## 已知约束

- Windows IME 兼容：块编辑用 `blur` 事件提交（`change` 会在输入法确认候选词时触发组件重建）
- transformers 建议 4.44–4.49（5.x 与 torch 2.5 存在 `CPUOffloadPolicy` 兼容问题）
- llama.cpp 模式：base 模型的 GGUF 通常未内嵌聊天模板，`chat` 上下文模式会报错提示改用 `prefix`/`raw`；逐 token 困惑度经 `llama_get_logits` 读取原始分布，接口不可用的老版本自动退化为无困惑度
- API 模式下上下文困惑度不可用（Chat Completions 不返回 prompt token 概率）；DeepSeek 等不支持 logprobs 的服务自动降级为无逐 token 困惑度

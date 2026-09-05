# 🖋 生成式文本编辑器

基于 LLM 的分块式生成文本编辑器：提示词块引导生成，生成块流式续写、可随时暂停与手动编辑。类 Jupyter 的交互体验，专为"人机协作写作"设计。

提供三种前端，共享同一套后端与块文档模型：

- **Web 块编辑器**（`web/`，`python server.py` 后浏览器访问 `http://127.0.0.1:8907/`）：**推荐**。以"类型化块数组"为唯一数据源，块边界是 UI 元素而非文本标记——所见即所得，后端零文本解析处理；带困惑度热力图与"LLM 视角预览"

- **Gradio 网页前端**（`app.py`，DEPRECATED）：浏览器内编辑，带全量困惑度可视化；已被 Web 块编辑器取代，保留作回退

- **VSCode 插件前端**（`vscode-extension/`）：以 WebviewPanel 编辑器标签页完整承载 Web 块编辑器——与浏览器版共用同一份 `web/` 前端代码，仅传输层与平台能力经 `web/platform.js` 抽象区分（webview 内所有请求由插件主进程 postMessage 代理，不直连网络）；命令面板执行 `GTE: 打开 Web 编辑器` 进入

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

- **llama.cpp 后端**（`llama-cpp-python`）：进程内加载 GGUF 量化模型（Q4\_K\_M / Q5\_K\_M / Q8\_0 / IQ 系列等），小显存 / 纯 CPU 也能跑大模型；支持 GPU offload 层数（`n_gpu_layers`）与上下文窗口（`n_ctx`）调节；聊天模板读取 GGUF 内嵌 `tokenizer.chat_template` 渲染（与 llama.cpp server 行为一致）；逐 token / 上下文困惑度可用；KV 前缀缓存由 `Llama.generate` 自动复用

- **API 后端**（OpenAI 兼容）：覆盖 OpenAI / DeepSeek / Qwen / GLM / Kimi，以及 vLLM / Ollama / llama.cpp server 等本地推理服务，SSE 流式

### 困惑度（Perplexity）分析

- **上下文困惑度**：一次前向传播评估提示词与模型的匹配度（本地 / llama.cpp 模式）

- **逐 token 生成困惑度**：本地用 LogitsProcessor 采集；API 模式经 logprobs 自动探测（不支持时降级）

- **prefill 困惑度**：生成前的上下文打分与前向同源，活动块内手动编辑的文本同样获得逐 token 真实着色（`prefill_ppl` 事件；API 模式不可用）

- **伪彩色热力图**：绿（模型确定）→ 红（模型困惑），覆盖**全部生成块**

- 折线图 + 滑动平均 + 全文档几何平均困惑度

- **编辑容错**：手动编辑后，仅编辑点之后的着色失效（灰显），编辑点之前的保留；新生成部分继续着色

### KV 前缀缓存（本地后端）

- 保存上次生成的 KV Cache，续写/再次生成时按最长公共前缀复用，仅对新增 token 增量 prefill

- 长文本场景首 token 延迟实测约 **1/3**（2.8x 加速），贪心输出与全量 prefill 完全一致

- 生成线程串行化保护，停止后立即重启生成不会产生缓存竞态

- 状态栏实时显示「⚡KV缓存复用 X/Y token」

### 上下文模式

| 模式       | 说明                              | 适用              |
| -------- | ------------------------------- | --------------- |
| `chat`   | 提示词块→user、生成块→assistant，经聊天模板包装 | Instruct 模型（默认） |
| `prefix` | 各块加【指令】/【正文】前缀后拼接               | base 模型         |
| `raw`    | 全部块原文裸拼接                        | 纯续写场景           |

### Agent Skill 接入

- 支持 Anthropic 风格 `SKILL.md`（YAML frontmatter + markdown 正文）与普通 `.md` 提示词文件

- 放入 `./skills/` 目录或直接上传，勾选后作为系统指令注入生成上下文

## 安装与运行

```bash
pip install -r requirements.txt

# HuggingFace 无法直连时可使用镜像
set HF_ENDPOINT=https://hf-mirror.com
```

### 方式〇：Web 块编辑器（推荐）

```bash
python server.py
# 浏览器打开 http://127.0.0.1:8907/
```

文档以 JSON 块数组保存于 `saves/*.json`（含逐 token 困惑度数据，刷新后着色完整还原）；旧 Markdown 文档可经「导入MD」按需迁移，亦可一键「导出MD」。

后端模式选择同下方 Gradio 说明（本地模型 / llama.cpp / API）。

### 方式一：Gradio 网页前端（DEPRECATED）

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

在顶栏选择后端模式：

- **本地模型**：填写模型路径或 HF ID（默认 `Qwen/Qwen2.5-0.5B-Instruct`），点击「加载模型」

- **llama.cpp（GGUF 量化模型）**：填写本地 `.gguf` 文件路径（如 `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`）或 HF GGUF 仓库 ID（如 `Qwen/Qwen2.5-0.5B-Instruct-GGUF`，首次自动下载），调节 `n_gpu_layers` / `n_ctx` 后点击「加载 GGUF」；需 `pip install llama-cpp-python`

- **API**：填写 base\_url（到 `/v1` 层级）、API Key、模型名，点击「连接 API」

### 方式二：VSCode 插件前端

插件以 WebviewPanel 编辑器标签页承载 Web 块编辑器，与浏览器版共用同一份前端代码（`web/`）。平台差异经 `web/platform.js` 抽象：webview 内所有 HTTP（含 SSE 生成流）由插件主进程 postMessage 代理，不直连网络；设置与未保存草稿经插件 globalState 持久化。

插件**自带 Python 无头服务端**（打包在扩展目录 `python/` 内），安装后无需手动启动、无需指定 `server.py` 路径——首次使用时插件会自动拉起服务。

1. 安装插件：在 VSCode 中「扩展」→「…」→「从 VSIX 安装」
   （源码调试则在 `vscode-extension/` 内 `npm install` 后按 F5；
   从源码打包 VSIX：`npx vsce package`，`python/` 服务端会自动从仓库根同步）

2. 确保本机 Python 已安装服务依赖（只需一次；跑本地模型还需 torch/transformers，跑 GGUF 量化模型还需 llama-cpp-python）：

   ```bash
   pip install fastapi uvicorn pyyaml requests
   pip install llama-cpp-python   # llama.cpp 模式（GGUF 量化模型）
   ```

3. 命令面板执行 `GTE: 打开 Web 编辑器`，在编辑器标签页中连接服务、加载模型 / 连接 API、勾选技能并调整生成参数；文档与浏览器版共用 `saves/*.json`，两边可互开同一份文档

> 若使用仓库根目录的 `server.py`（例如调试最新改动），可在设置 `gte.serverScript` 中指定其绝对路径，插件将优先使用。

#### 文档格式

文档以 JSON 块数组保存在服务端 `saves/*.json`（含逐 token 困惑度数据）；Markdown 仅作「导入MD」/「导出MD」的交换格式。交换格式中提示词块是 `<!-- prompt ... -->` 注释（渲染不可见），生成块是可见正文，每块生成正文前有一行 `<!-- generate -->` 注释作为块边界标记（用于分隔相邻生成块、标记空的活动生成单元）：

```markdown
<!-- prompt
写一段关于秋天的散文开头，100字左右。
-->

秋日的午后，阳光穿过梧桐叶洒在青石板上……

<!-- generate -->

（定稿后新开的活动生成块，生成从这里继续）
```

提示词注释也支持单行写法 `<!-- prompt: 指令内容 -->`。生成仅发生在**最后一块生成块**（活动生成单元）中：文档末尾没有生成块时按 `Ctrl+Enter` 会自动创建。

**系统提示词块**（`<!-- system ... -->` 注释）：技能可经技能库列表的「固化」按钮（Gradio 端为「📥 固化选中技能为系统块」按钮）固化为文档顶部的系统块，生成时作为系统级指令拼入上下文。固化后文档自包含——换到没有该技能的环境（另一台机器 / 另一个 skills 目录）也能完整复现生成过程：

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

**思维链块**（`<!-- cot ... -->` 注释）：推理模型（DeepSeek-R1 / GLM / Qwen3 等）生成时，思考过程与正文分离——思维链流式写入活动生成块前紧邻的 cot 注释块（Markdown 渲染中不可见，源码中可查看/编辑），正文照常进入生成块。三后端均支持：API 模式捕获 `reasoning_content` 流；本地/llama.cpp 模式解析内嵌 `<think>` … `<<arg_key:6124c78e>>` 标签。**cot 块自包含完整思考区**（`<think>` 与闭标签直接写在注释里，所见即所得）：

```markdown
<!-- prompt
写一段关于秋天的散文开头，100字左右。
-->

<!-- cot
<think>
（模型思考过程，随文档保存、渲染不可见）
<<arg_key:6124c78e>>
-->

秋日的午后，阳光穿过梧桐叶洒在青石板上……
```

**中断后续写的行为由 cot 的闭合状态决定**：保留闭标签 `<<arg_key:6124c78e>>` → 思考已完成，再次生成模型基于已有思考**直接输出正文**；删掉闭标签（思考被中断未闭合）→ 再次生成模型**从断点继续思考**。用户编辑标签即可控制，无需额外设置。模型自然结束（输出过闭标签）时插件端自动补写闭标签；手动停止或思考被截断时保持未闭合。

**临近思维链恒回灌**：活动生成单元前紧邻的 cot 块属于"当前进行中的 assistant 回复"，无论正文是否已开始，每次生成都作为思考区回灌进 prompt（历史轮次的思维链仍不进上下文）。这保证续写上下文与首次生成一致——已生成文字的困惑度着色不漂移，KV 前缀缓存可整体复用。

> 注意：Qwen3.5-4B 在未闭合回灌时会持续思考较久（不主动闭合），想尽快出正文可手动补上闭标签或按 `F2` 定稿开新块。回灌会把思维链写进 prompt，prefill 开销随其长度线性增长（本地/llama.cpp 后端已利用 KV 前缀复用，仅新增部分 prefill）。

Gradio 端在生成块上方提供「🧠 思维链（当前轮）」流式单元格，定稿时随块保存为 cot 块；历史思维链块默认折叠，展开可编辑。

**思维链能力探测与思考模式开关**：模型加载后前端显示思维链能力（✅ 支持 / ❓ 待检测，`❓` 在首次生成检测到思维链后自动转为 ✅；本地/llama.cpp 模式依据聊天模板是否含 `enable_thinking` 变量、模型名是否命中推理系关键词推断）。生成参数提供「🧠 思考模式」三态开关（auto/on/off）：本地与 llama.cpp 模式均经聊天模板 `enable_thinking` 变量生效（Qwen3 系），API 模式发送 `chat_template_kwargs`（vLLM/SGLang 约定，服务端不认识时自动降级）。

> llama.cpp 模式能否开关取决于 GGUF 内嵌模板是否识别 `enable_thinking`：Qwen3 系模板在该变量为真时渲染出**未闭合**的 `<think>`，生成流因此不含开标签、只含闭标签，分离器按「思考正文自流首开始」处理；模板不识该变量时开关无效，思考与否由模板默认行为决定。

**思维链困惑度**：本地/llama.cpp 后端思考区 token 的 log-prob 与正文同源，思维链块同样按绿→红色阶着色（前端困惑度面板与指标；定稿后随块冻结）；API 后端不返回 reasoning 的 logprobs，思维链不着色。

#### 命令与快捷键

插件仅保留一个入口命令；编辑操作在 Web 编辑器（webview）内完成，快捷键由编辑器自身捕获，与浏览器版一致：

| 命令 / 快捷键           | 说明                                  |
| ------------------ | ----------------------------------- |
| `GTE: 打开 Web 编辑器`  | 打开（或聚焦已打开的）块编辑器标签页；点击状态栏 `GTE:` 项同效 |
| `Ctrl+Enter`（编辑器内） | 生成 / 停止（同一按键切换）                     |
| `F2`（编辑器内）         | 定稿当前生成块并开启新块                        |
| `Ctrl+S`（编辑器内）     | 保存文档                                |

旧命令（`GTE: 生成 / 停止`、`GTE: 定稿当前块`、`GTE: 追加提示词块`、`GTE: 固化技能为系统提示词块`、`GTE: 锁定/解锁光标所在块`、`GTE: 新建生成式文本文档` 等）随「Markdown 源码 + 装饰」旧模式一并移除，功能由 Web 编辑器内对应按钮/快捷键承接；`GTE: 打开生成控制面板` 保留为 `GTE: 打开 Web 编辑器` 的别名。

流式生成的 token 按困惑度着色（绿=确定、红=困惑）；手动编辑后颜色先灰显回退，下次生成时 prefill 打分会为编辑过的文本重新着色（本地 / llama.cpp 模式）。

#### 插件配置

| 配置项                   | 默认值                     | 说明                            |
| --------------------- | ----------------------- | ----------------------------- |
| `gte.serverUrl`       | `http://127.0.0.1:8907` | 无头服务地址                        |
| `gte.autoStartServer` | `true`                  | 服务不可达时自动用 python 拉起 server.py |
| `gte.pythonCommand`   | `python`                | 启动 server.py 所用 Python 命令     |
| `gte.serverScript`    | 扩展内 `python/`           | server.py 绝对路径（留空用随包自带服务端）    |

生成参数与技能勾选在 Web 编辑器内调整，持久化于插件 globalState（浏览器版则存 localStorage），不再经 `gte.params` 配置。

## 项目结构

```
├── web/             # Web 块编辑器前端（纯 vanilla JS，server.py 托管）：
│   ├── model.js     #   块模型、ppl 对账、LLM 视角消息映射（可移植内核）
│   ├── editor.js    #   块渲染、contenteditable、块操作、流式更新（可移植内核）
│   ├── platform.js  #   平台抽象：浏览器 / VSCode webview 双实现（可移植内核）
│   ├── api.js       #   REST + SSE 客户端（浏览器平台实现复用）
│   ├── app.js       #   编排：文档管理、面板、生成控制器、快捷键
│   ├── webview-main.js # VSCode webview 打包入口（esbuild → dist/webview.js）
│   ├── index.html / style.css
├── app.py            # Gradio 前端（DEPRECATED，保留作回退）
├── server.py         # FastAPI 服务：REST + SSE + 文档 CRUD + Web 前端托管
├── core.py           # 共享纯逻辑：文档序列化、上下文组装、困惑度聚合
├── backend.py        # LLM 后端：LocalBackend（KV缓存）/ LlamaCppBackend（GGUF 量化）/ OpenAICompatBackend（SSE）
├── skills.py         # 技能扫描、解析（SKILL.md frontmatter）、上下文拼接
├── skills/           # 技能库目录
│   └── 中文散文写作/SKILL.md
├── saves/            # 文档保存目录（运行时生成）
├── vscode-extension/ # VSCode 插件前端（WebviewPanel 承载 web/ 块编辑器）
│   ├── src/          # 插件源码（TS）：extension（入口）/ webEditor（Webview 代理）/ server / api
│   ├── python/       # 构建产物：打包时从仓库根自动复制（不入 git）
│   ├── package.json  # 命令、配置项声明
│   └── esbuild.js    # 构建/打包脚本（extension + webview 双入口 + 同步 python/）
├── requirements.txt
├── kv_cache_test.py      # KV 缓存回归：命中加速/贪心一致性/停止后再生成
├── ppl_accum_test.py     # 困惑度累积回归：跨块/编辑容错/序列化兼容
├── lock_autosave_test.py # 块锁定 + 定时自动保存逻辑回归
├── regress3_test.py      # 服务器 REST 回归（需 app.py 已启动）
├── server_test.py        # 无头服务 REST/SSE 回归（需 server.py 已启动）
├── test_llamacpp_mock.py # LlamaCppBackend mock 单测（无需模型/llama-cpp-python）
├── test_backend_contract.py # 三后端统一 GenUpdate 契约测试（local 需 venv 含 torch）
└── test_llamacpp_model.py # llama.cpp 真实模型回归（GGUF_MODEL 指定模型，缺失时跳过）
```

## 后端行为契约

三后端（local / llamacpp / api）共享同一套接口（`loaded / load / build_chat_prompt / build_flat_prompt / compute_context_ppl / generate_stream / stop / kind`，签名一致），
`generate_stream` 产出的 `GenUpdate` 流还须满足统一不变式（前端 `web/editor.js`
的困惑度着色对账依赖——浏览器与 VSCode webview 共用同一份代码，详见
`tests/test_backend_contract.py`）：

- `final=True` 的快照有且只有最后一个；

- `token_texts` / `token_ppls`（及 reasoning 对应字段）一一等长；

- `join(token_texts)` 恒为 `cum_text` 前缀、final 时相等；

- `join(reasoning_token_texts)` 恒为 `reasoning_cum` 前缀、final 时相等，且
  **不含思考区标签字符**（仅思考正文）——曾因标签混入导致 transformers 后端
  思维链整块不着色（正文正常），llama.cpp 后端无此问题，正是两类后端行为
  不一致的回归锚点；

- API 后端无 reasoning logprobs → `reasoning_token_texts` 恒空（已知降级，非失配）；

- local 后端 think 态流式期间也逐 token 对齐（未闭合思考同样着色），与 llama.cpp 行为一致。

## 已知约束

- Windows IME 兼容：块编辑用 `blur` 事件提交（`change` 会在输入法确认候选词时触发组件重建）

- transformers 建议 4.44–4.49（5.x 与 torch 2.5 存在 `CPUOffloadPolicy` 兼容问题）

- llama.cpp 模式：base 模型的 GGUF 通常未内嵌聊天模板，`chat` 上下文模式会报错提示改用 `prefix`/`raw`；逐 token 困惑度经 `llama_get_logits` 读取原始分布，接口不可用的老版本自动退化为无困惑度

- API 模式下上下文困惑度不可用（Chat Completions 不返回 prompt token 概率）；DeepSeek 等不支持 logprobs 的服务自动降级为无逐 token 困惑度


# 🖋 生成式文本编辑器

基于 LLM 的分块式生成文本编辑器：提示词块引导生成，生成块流式续写、可随时暂停与手动编辑。类 Jupyter 的交互体验，专为"人机协作写作"设计。

提供两种前端，共享同一套后端与文档模型：
- **Gradio 网页前端**（`app.py`）：浏览器内编辑，带全量困惑度可视化
- **VSCode 插件前端**（`vscode-extension/`）：在 Markdown 里用 `<prompt>` / `<generate>` 标签写作，习惯 VSCode 编辑的用户首选

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
```

### 方式一：Gradio 网页前端

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

在顶栏选择后端模式：
- **本地模型**：填写模型路径或 HF ID（默认 `Qwen/Qwen2.5-0.5B-Instruct`），点击「加载模型」
- **API**：填写 base_url（到 `/v1` 层级）、API Key、模型名，点击「连接 API」

### 方式二：VSCode 插件前端

插件**自带 Python 无头服务端**（打包在扩展目录 `python/` 内），安装后无需手动启动、无需指定 `server.py` 路径——首次打开面板时插件会自动拉起服务。

1. 安装插件：在 VSCode 中「扩展」→「…」→「从 VSIX 安装」
   （源码调试则在 `vscode-extension/` 内 `npm install` 后按 F5）

2. 确保本机 Python 已安装服务依赖（只需一次；跑本地模型还需 torch/transformers）：

   ```bash
   pip install fastapi uvicorn pyyaml requests
   ```

3. 在侧边栏 **GTE 生成控制面板** 中连接服务、加载模型 / 连接 API、勾选技能并调整生成参数

> 若使用仓库根目录的 `server.py`（例如调试最新改动），可在设置 `gte.serverScript` 中指定其绝对路径，插件将优先使用。

#### 文档格式

插件以 Markdown 为文档载体，`<prompt>` 与 `<generate>` 标签划分块：

```markdown
<prompt>
写一段关于秋天的散文开头，100字左右。
</prompt>

<generate>
秋日的午后，阳光穿过梧桐叶洒在青石板上……
</generate>
```

生成仅发生在**最后一块生成块**（活动生成单元）中：文档末尾没有 `<generate>` 块时按 `Ctrl+Enter` 会自动创建。

#### 命令与快捷键

| 命令 | 快捷键 | 说明 |
|---|---|---|
| `GTE: 生成（续写当前生成块）` | `Ctrl+Enter` | 从光标后开始流式生成 |
| `GTE: 停止生成` | `Ctrl+Alt+Enter` | 停止当前生成，保留已生成文本 |
| `GTE: 定稿当前块并开启新块` | `Ctrl+Shift+Enter` | 锁定前序块、开启新的活动生成块 |
| `GTE: 追加提示词块` | — | 在文档末尾追加 `<prompt>` 块 |
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
├── backend.py        # LLM 后端：LocalBackend（KV缓存）/ OpenAICompatBackend（SSE）
├── skills.py         # 技能扫描、解析（SKILL.md frontmatter）、上下文拼接
├── skills/           # 技能库目录
│   └── 中文散文写作/SKILL.md
├── saves/            # 文档保存目录（运行时生成）
├── vscode-extension/ # VSCode 插件前端
│   ├── src/          # 插件源码（TS）：extension/generation/docmodel/panel/… 
│   ├── package.json  # 命令、快捷键、配置项声明
│   └── esbuild.js    # 打包脚本
├── requirements.txt
├── kv_cache_test.py      # KV 缓存回归：命中加速/贪心一致性/停止后再生成
├── ppl_accum_test.py     # 困惑度累积回归：跨块/编辑容错/序列化兼容
├── lock_autosave_test.py # 块锁定 + 定时自动保存逻辑回归
├── regress3_test.py      # 服务器 REST 回归（需 app.py 已启动）
└── server_test.py        # 无头服务 REST/SSE 回归（需 server.py 已启动）
```

## 已知约束

- Windows IME 兼容：块编辑用 `blur` 事件提交（`change` 会在输入法确认候选词时触发组件重建）
- transformers 建议 4.44–4.49（5.x 与 torch 2.5 存在 `CPUOffloadPolicy` 兼容问题）
- API 模式下上下文困惑度不可用（Chat Completions 不返回 prompt token 概率）；DeepSeek 等不支持 logprobs 的服务自动降级为无逐 token 困惑度

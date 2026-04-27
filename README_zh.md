# FinAgent Workbench

[English](./README.md) | 中文

FinAgent Workbench 是一个基于 [DeerFlow](https://github.com/bytedance/deer-flow) 和 [PIXIU](https://github.com/The-FinAI/PIXIU) 的 agentic financial analysis 项目。项目使用 DeerFlow 作为 agent 编排和工作台层，使用 PIXIU/FinMA 作为金融领域专家模块，用于短上下文金融推理任务。

这个项目的关键设计是：**PIXIU/FinMA 不作为主控 agent 模型**。FinMA v0.1 基于较早期的 LLaMA 系模型，上下文窗口较短，不适合做长链路 agent 编排、多来源综合、完整财报分析和最终报告生成。FinAgent 使用 GLM、DeepSeek、Qwen 等长上下文通用模型作为 lead model，把 FinMA 路由为金融专家模块。

## 项目概览

```text
User
  -> FinAgent Workbench UI
  -> DeerFlow lead agent
       -> 长上下文模型：GLM / DeepSeek / Qwen
       -> 工具：
            - Web search and fetch
            - 财务数据检索
            - 财务计算器
            - FinMA expert module
       -> Skills：
            - financial analysis
            - earnings analysis
            - valuation
            - risk review
       -> 输出：
            - 投资备忘录
            - 财报分析
            - 同业比较
            - 风险备忘录
            - 评测报告
```

## 为什么这样设计

金融分析同时需要两类能力：

- 长上下文编排能力：阅读财报、收集新闻、协调工具、维护任务状态、生成最终报告。
- 金融领域判断能力：情绪分类、风险识别、事件影响分析、金融信号抽取、短文本金融问答。

现代长上下文模型更适合第一类能力。PIXIU/FinMA 更适合第二类能力。因此 FinAgent 采用混合架构，而不是强行让一个模型承担全部任务。

## 核心工作量

项目工作量分为系统工程、模型训练和评测三部分。

### 1. Agent 系统

- 将 DeerFlow 改造成金融分析工作台。
- 增加金融任务预设和报告工作流。
- 定义金融分析 skills。
- 增加 lead model 与金融专家模块之间的模型路由。
- 增加财报、新闻、行情和财务计算工具。

### 2. FinMA 专家模块

- 将 PIXIU/FinMA 包装成可被 DeerFlow 调用的金融专家模块。
- 输入保持短上下文和任务特定。
- 输出结构化结果，例如标签、理由、置信度和抽取出的金融信号。
- GPU endpoint 未就绪时，先使用 mock tool。
- 后续将 mock 替换为真实 vLLM 或 OpenAI-compatible FinMA 服务。

### 3. 后训练

- 使用 LoRA 或 QLoRA 对 FinMA 进行后训练。
- 训练目标聚焦短上下文金融专家任务，而不是长上下文 agent planning。
- 候选任务：
  - 金融情绪分类
  - 风险因素分类
  - 事件影响分析
  - 管理层语气分析
  - 短段落金融问答
  - 结构化金融信号抽取

### 4. 评测

对比对象：

- GLM only
- DeepSeek only
- original FinMA on supported short tasks
- GLM + original FinMA module
- GLM + post-trained FinMA module
- DeepSeek + post-trained FinMA module

评测内容：

- PIXIU 的 FinBen 任务
- 自定义端到端金融分析任务
- 报告质量评分
- 引用准确性
- 延迟和成本
- 失败案例分析

## 当前状态

已完成：

- DeerFlow 仓库已拉取并本地跑通。
- 本地前端入口已统一到 `http://localhost:2026/`。
- 前端已改造成 FinAgent Workbench。
- 已增加金融 landing page。
- 已增加金融聊天预设。
- 已起草 PRD。

计划中：

- 新增 `financial-analysis` skill。
- 新增 `finma_expert` mock tool。
- 通过 vLLM 或 OpenAI-compatible API 接入真实 FinMA endpoint。
- 构建小规模评测集。
- 准备后训练数据和 LoRA/QLoRA 脚本。

## 本地开发

### 环境要求

- Python 3.12+
- Node.js 22+
- pnpm
- uv
- nginx

当前本地环境中，Node.js 22 通过 Homebrew 的 `node@22` 安装，因此运行命令时需要带上：

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH
```

### 启动应用

在仓库根目录执行：

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH make stop
screen -dmS deerflow zsh -lc 'cd /Users/yuk1no/6052/deer-flow && PATH=/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin ./scripts/serve.sh --dev --skip-install'
```

打开：

```text
http://localhost:2026/
```

### 停止应用

```bash
screen -S deerflow -X quit
PATH=/opt/homebrew/opt/node@22/bin:$PATH make stop
```

### 验证

```bash
curl -I http://localhost:2026/
```

预期结果：

```text
HTTP/1.1 200 OK
```

## 模型接入计划

### Lead Model

使用长上下文模型负责 orchestration：

```yaml
models:
  - name: glm-lead
    display_name: GLM Lead Model
    use: langchain_openai:ChatOpenAI
    model: your-glm-model
    api_key: $GLM_API_KEY
    base_url: https://your-glm-compatible-endpoint/v1
    request_timeout: 600.0
    max_retries: 2
```

### FinMA Expert Endpoint

将 FinMA 暴露成单独的专家服务：

```text
DeerFlow tool call
  -> finma_expert endpoint
  -> FinMA model
  -> structured financial judgment
```

专家服务输入应保持短上下文：

```json
{
  "task": "risk_classification",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth but faces export control restrictions.",
  "schema": "risk_type,impact_direction,confidence,rationale"
}
```

预期输出：

```json
{
  "risk_type": "regulatory_risk",
  "impact_direction": "negative",
  "confidence": 0.78,
  "rationale": "Export controls can limit sales in specific markets."
}
```

## 文档

- [English PRD](./docs/FINAGENT_PRD.md)
- [中文 PRD](./docs/FINAGENT_PRD_zh.md)
- [本地设置记录](./docs/FINANCIAL_AGENT_SETUP.md)

## 基于以下开源项目

本项目基于以下开源项目构建：

- [DeerFlow](https://github.com/bytedance/deer-flow)：agent 编排、工作台 UI、工具、skills、memory、sandbox 和 sub-agent runtime。
- [PIXIU](https://github.com/The-FinAI/PIXIU)：金融大模型、金融指令数据和 FinBen 评测基准。

## 引用

如果使用 PIXIU/FinMA，请引用 PIXIU 论文：

```bibtex
@misc{xie2023pixiu,
  title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance},
  author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
  year={2023},
  eprint={2306.05443},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

如果使用 DeerFlow，请引用或标注上游项目：

```text
https://github.com/bytedance/deer-flow
```

## License

除项目维护者后续另行调整外，本仓库沿用原 DeerFlow MIT License。详见 [LICENSE](./LICENSE)。


# FinAgent Workbench PRD

## 1. 项目概述

FinAgent Workbench 是一个基于 DeerFlow 和 PIXIU 的 agentic financial analysis 系统。系统使用长上下文通用大模型作为 orchestration model，使用 PIXIU/FinMA 作为金融领域专家模块，处理短上下文金融推理任务。

本项目的产品目标不是让 FinMA 成为全局 agent brain。FinMA v0.1 的上下文窗口较短，不适合长链路规划、多来源综合、完整财报分析和最终报告生成。FinAgent 将 FinMA 定位为一个可训练、可评测、可对比的 specialist module，并把它嵌入到更大的 agent workflow 中。

## 2. 问题定义

金融分析任务同时需要长上下文编排能力和金融领域判断能力。

长上下文编排能力用于：

- 阅读长财报、earnings transcript、新闻集合和工具输出。
- 维护多步骤 agent 状态。
- 协调检索、计算、批判和报告生成。
- 生成带引用的结构化投资备忘录。

金融领域判断能力用于：

- 金融情绪分类。
- 风险因素识别。
- 事件影响分析。
- 管理层语气分析。
- 金融术语解释。
- 短上下文金融问答。

PIXIU/FinMA 对第二类任务有价值，但上下文长度限制使它不适合第一类任务。因此系统架构必须明确这两类能力的边界。

## 3. 产品定位

FinAgent Workbench 是一个混合式 agentic financial analysis 平台：

- DeerFlow 提供 agent runtime、UI、工具执行、skills、memory、artifacts 和 sub-agent orchestration。
- GLM、DeepSeek、Qwen 或其他长上下文模型作为 lead agent。
- PIXIU/FinMA 作为 financial expert module。
- 金融工具提供财报、新闻、行情数据和计算能力。
- FinBen 和自定义任务提供评测闭环。

推荐项目描述：

> Agentic Financial Analysis with Long-Context Orchestration and Domain-Tuned Financial Expert Modules.

短版本：

> Agentic Financial Analysis with Domain-Tuned LLM Modules.

## 4. 目标

### 4.1 产品目标

- 提供 Web 端金融分析工作台。
- 支持用户请求财报分析、估值备忘录、风险审查、同业比较、市场简报和组合诊断。
- 生成结构清晰、带来源引用的投研输出。
- 支持 lead model 和 financial expert model 之间的模型路由。
- 支持后续对 FinMA expert module 的训练和评测。

### 4.2 研究目标

- 证明 domain-tuned financial LLM 即使不适合做 lead agent，也可以作为专家模块产生价值。
- 比较 general LLM-only workflow 和 hybrid workflow。
- 比较原始 FinMA 和后训练 FinMA。
- 围绕 agent orchestration、tool integration、post-training 和 evaluation 构建可展示的 workload。

### 4.3 非目标

- 不强行让 FinMA 做 DeerFlow 主模型。
- 不声称 FinMA 可以端到端处理长金融文档。
- 不构建生产级投资顾问。
- 不在缺少风险提示和来源约束时提供投资建议。
- 不把 UI 改造作为唯一项目贡献。

## 5. 目标用户

### 5.1 主要用户

- 构建 agentic financial analysis 项目的学生或研究者。
- 评估金融领域大模型的开发者。
- 需要结构化金融文本摘要的分析人员。

### 5.2 次要用户

- 评估系统的老师或 reviewer。
- 对比模型表现的研究组成员。
- 测试端到端金融 workflow 的 demo 用户。

## 6. 核心用户场景

### 场景 1：财报分析

用户输入：

```text
Analyze Apple's latest earnings. Cover revenue drivers, margin trend, cash flow, guidance, catalysts, risks and source-backed conclusions.
```

系统行为：

- Lead agent 拆解任务。
- 检索工具收集财报、earnings news 和市场上下文。
- 财务计算器抽取或计算关键指标。
- FinMA expert module 对短文本 chunk 做情绪、风险、事件影响和金融信号判断。
- Lead agent 综合生成最终报告。

预期输出：

- Executive Summary
- Business and Revenue Drivers
- Financial Performance
- Management Tone
- Catalysts
- Risks
- Investment View
- Sources

### 场景 2：同业比较

用户输入：

```text
Compare NVIDIA and AMD across growth, margins, valuation, competitive position, catalysts and risks.
```

系统行为：

- Lead agent 收集两家公司的数据。
- 工具标准化财务指标。
- FinMA expert module 评估新闻、transcript 和 filing 中的短片段。
- Lead agent 输出比较表和结论。

### 场景 3：风险审查

用户输入：

```text
Review Tesla's main investment risks, including accounting, liquidity, regulation, competition and macro sensitivity.
```

系统行为：

- Lead agent 收集 risk factor disclosures 和新闻。
- FinMA expert module 分类风险类型和影响方向。
- Lead agent 生成排序后的风险 memo。

### 场景 4：模型对比实验

研究者选择 benchmark task 并运行：

- GLM only。
- DeepSeek only。
- GLM + original FinMA module。
- GLM + post-trained FinMA module。
- DeepSeek + post-trained FinMA module。

预期输出：

- 准确率或评分指标。
- 延迟和成本。
- 错误类型。
- 定性案例。

## 7. 系统架构

```text
User
  -> FinAgent Workbench UI
  -> DeerFlow Lead Agent
       -> Long-context model: GLM / DeepSeek / Qwen
       -> Tools:
            - Web search
            - Web fetch
            - Filings/news/market data retrieval
            - Financial calculator
            - FinMA expert module
       -> Skills:
            - financial-analysis
            - earnings-analysis
            - valuation
            - risk-review
       -> Artifacts:
            - investment memo
            - metric tables
            - citations
            - evaluation reports
```

## 8. 模型策略

### 8.1 Lead Model

Lead model 负责：

- 长上下文。
- 工具规划。
- Agent orchestration。
- 多步骤推理。
- 报告综合。
- 用户交互。

候选 lead models：

- GLM long-context model。
- DeepSeek model。
- Qwen long-context model。
- 其他 OpenAI-compatible 长上下文模型。

选择标准：

- 上下文长度。
- 工具调用稳定性。
- 成本。
- 延迟。
- 中英文支持。
- 报告质量。

### 8.2 FinMA Expert Module

FinMA 负责短上下文金融任务：

- 新闻情绪分类。
- 风险类型分类。
- 事件影响方向判断。
- 管理层语气分析。
- 金融实体和关系抽取。
- 金融术语解释。
- 短文本金融 QA。

输入应切块并限制长度。建议输入大小：

- 每次调用 512 到 1500 tokens。
- 只包含任务相关上下文。
- 不发送完整财报或完整对话历史。

输出尽量采用结构化 JSON：

```json
{
  "task": "risk_classification",
  "label": "regulatory_risk",
  "impact_direction": "negative",
  "confidence": 0.78,
  "rationale": "The passage discusses export restrictions that may limit revenue growth."
}
```

### 8.3 Model Router

Model router 决定每个子任务由哪个模型处理。

Lead model 处理：

- 规划。
- 检索。
- 长文档综合。
- 最终报告写作。
- 多来源比较。

FinMA 处理：

- 短文本金融分类。
- 金融信号抽取。
- 领域解释。
- 后训练金融专家评测。

## 9. 功能需求

### 9.1 金融工作台 UI

状态：已部分实现。

需求：

- 展示 FinAgent 品牌。
- 提供金融任务 presets。
- 提供 starter prompts。
- 将 starter prompts 自动带入 chat input。
- 保持与 DeerFlow workspace 的兼容性。

### 9.2 Financial Analysis Skill

状态：计划中。

需求：

- 定义标准金融报告结构。
- 定义证据和引用要求。
- 定义何时调用 FinMA expert module。
- 定义风险和不确定性的处理方式。
- 定义以下输出模板：
  - 财报分析。
  - 估值 memo。
  - 同业比较。
  - 风险审查。
  - 市场简报。

### 9.3 FinMA Expert Tool

状态：计划中。

需求：

- 将 FinMA 暴露为 DeerFlow 可调用工具。
- GPU 未就绪时先使用 mock implementation。
- 后续接入真实 FinMA API。
- 接收 task type 和短上下文。
- 返回结构化 JSON。

示例 tool request：

```json
{
  "task": "event_impact",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth...",
  "schema": "impact_direction,rationale,confidence"
}
```

示例 tool response：

```json
{
  "impact_direction": "positive",
  "affected_factors": ["revenue_growth", "margin_expectation"],
  "confidence": 0.82,
  "rationale": "The passage indicates strong demand in data center revenue."
}
```

### 9.4 金融数据工具

状态：计划中。

MVP 工具：

- 使用 DeerFlow 现有 web search 和 web fetch。
- 简单行情数据检索。
- 财报检索或用户上传财报。
- Python financial calculator。

后续工具：

- SEC EDGAR integration。
- Yahoo Finance 或 Polygon integration。
- News API integration。
- Earnings call transcript retrieval。
- FinBen evaluation runner。

### 9.5 报告生成

状态：计划中。

报告应包括：

- Executive Summary。
- Company Overview。
- Financial Performance。
- Valuation。
- Catalysts。
- Risks。
- Investment View。
- Sources。

报告必须区分：

- 来自检索源的事实。
- 模型推断。
- 假设。
- 缺失或未知数据。

## 10. 后训练计划

### 10.1 训练目标

不要把 FinMA 训练成长上下文 lead agent。目标是把它训练成更强的短上下文金融专家。

目标能力：

- 金融情绪分类。
- 风险因素分类。
- 事件影响分析。
- 管理层语气分析。
- 短文本金融 QA。
- 结构化金融信号抽取。

### 10.2 训练方法

推荐方法：

- LoRA 或 QLoRA。
- Base：FinMA-7B-full 或 FinMA-7B-NLP。
- 训练框架：Hugging Face Transformers + PEFT。
- 硬件：量化训练可单卡，完整训练需要多卡。

### 10.3 训练数据

数据来源：

- PIXIU FIT instruction data。
- 转换为 instruction format 的 FinBen tasks。
- 自定义金融分析数据。
- 公开财报和 earnings call snippets。
- 经过验证的合成金融分类样本。

推荐 instruction format：

```json
{
  "instruction": "Classify the financial risk type and explain the impact.",
  "input": "The company warned that export controls may reduce revenue in China...",
  "output": {
    "risk_type": "regulatory_risk",
    "impact_direction": "negative",
    "confidence": 0.86,
    "rationale": "Export controls can restrict sales in a key market."
  }
}
```

### 10.4 训练版本

至少运行两个版本：

- FinMA-original：不做额外训练。
- FinMA-LoRA-financial-signals：针对短上下文金融分类和抽取任务训练。

可选：

- FinMA-LoRA-report-style：训练短上下文下的结构化 analyst-style 输出。

## 11. 评测计划

### 11.1 Baselines

评测对象：

- GLM only。
- DeepSeek only。
- Original FinMA only on supported short tasks。
- GLM + original FinMA module。
- GLM + post-trained FinMA module。
- DeepSeek + post-trained FinMA module。

### 11.2 Benchmarks

使用两组 benchmark。

Group A：标准金融 NLP 任务：

- FinBen classification tasks。
- Sentiment tasks。
- Headline classification。
- Named entity recognition。
- Short QA。

Group B：Agentic financial analysis 任务：

- Earnings analysis。
- Risk review。
- Peer comparison。
- Market brief。
- Investment memo generation。

### 11.3 Metrics

分类和抽取任务：

- Accuracy。
- F1。
- Macro-F1。
- JSON validity。
- Calibration 或 confidence quality。

报告生成任务：

- 事实准确性。
- 引用正确性。
- 必要章节覆盖率。
- 风险识别质量。
- 金融推理质量。
- 人工评分。

系统性能：

- 延迟。
- 成本。
- 工具调用次数。
- 每份报告调用 FinMA 的次数。
- 失败率。

### 11.4 预期假设

预期结果：

- 长上下文通用模型更适合做 lead agent。
- 原始 FinMA 在短金融 NLP 任务上有竞争力。
- 后训练 FinMA 能改善短上下文金融专家输出。
- Hybrid workflow 相比 lead-model-only workflow 可以提升金融信号质量。

## 12. MVP 范围

### 12.1 MVP Must Have

- FinAgent-branded DeerFlow UI。
- 可运行的本地 DeerFlow 服务。
- 金融 prompt presets。
- `financial-analysis` skill。
- Mock `finma_expert` tool，返回结构化 JSON。
- 至少一个完整 workflow：
  - 用户请求财报分析；
  - lead agent 收集上下文；
  - 调用 FinMA expert module；
  - 生成结构化报告。

### 12.2 MVP Should Have

- GPU 可用时接入真实 FinMA API。
- 简单市场数据检索。
- 报告导出为 Markdown。
- 小规模 benchmark evaluation script。

### 12.3 MVP Could Have

- SEC filing retrieval。
- News API integration。
- 模型对比结果 dashboard。
- 后训练 LoRA model endpoint。

## 13. 里程碑

### Milestone 1：产品外壳

交付物：

- FinAgent UI。
- 本地 DeerFlow service on port 2026。
- 金融 starter prompts。
- 架构说明文档。

状态：

- 基本完成。

### Milestone 2：金融工作流

交付物：

- `financial-analysis` skill。
- 报告模板。
- Prompting rules。
- 示例生成报告。

### Milestone 3：FinMA Expert Module

交付物：

- Mock FinMA expert tool。
- Tool schema。
- 调用 FinMA 的 routing rule。
- 使用 mock FinMA 的示例 workflow。

### Milestone 4：真实模型接入

交付物：

- 通过 vLLM 或兼容 API 部署 FinMA。
- DeerFlow tool 调用真实 FinMA endpoint。
- 延迟和失败日志。

### Milestone 5：后训练

交付物：

- 训练数据集。
- LoRA/QLoRA script。
- Trained adapter。
- Model card 和 training notes。

### Milestone 6：评测

交付物：

- FinBen task evaluation。
- Custom agentic benchmark。
- Baseline comparison table。
- Error analysis。

## 14. 技术风险

### Risk 1：FinMA 上下文长度

问题：

- FinMA 不适合长上下文 lead-agent use。

缓解：

- FinMA 只作为短上下文 expert module。
- 使用 chunking 和结构化任务 prompt。

### Risk 2：GPU 可用性

问题：

- FinMA-7B 需要 GPU 资源。

缓解：

- 先使用 mock tool。
- 后续在 GPU 上部署真实 FinMA。
- DeerFlow 本地运行，模型远端运行。

### Risk 3：FinMA 生成能力弱于新模型

问题：

- FinMA v0.1 较旧，可能弱于较新的 general models。

缓解：

- 在短金融任务上评测它，而不是在长文档生成上评测它。
- 使用 LoRA 做后训练。
- 对比 original 和 post-trained 版本。

### Risk 4：金融能力过度承诺

问题：

- 金融分析输出可能被误认为投资建议。

缓解：

- 增加免责声明。
- 强制要求来源。
- 区分事实、假设和模型推断。

### Risk 5：Benchmark 不匹配

问题：

- FinBen 不一定衡量端到端 agent quality。

缓解：

- FinBen 用于模块级评测。
- 自定义任务用于 workflow-level evaluation。

## 15. 成功标准

项目成功标准：

- DeerFlow 可以端到端运行金融分析 workflow。
- 系统清楚展示长上下文 orchestration + domain expert routing。
- FinMA 被集成为 expert module。
- 后训练 FinMA 可以和原始 FinMA 对比。
- 实验能够展示 GLM、DeepSeek、original FinMA module、post-trained FinMA module 的可测差异。
- 最终报告结构化、可追溯、有来源依据。

## 16. 近期下一步

1. 创建 `financial-analysis` skill。
2. 定义 `finma_expert` tool schema。
3. 实现 mock `finma_expert` endpoint 或 Python tool。
4. 配置 DeerFlow 在金融 workflow 中调用 mock expert。
5. 准备 20 到 50 条小规模评测样本。
6. 决定真实 FinMA 的 GPU 部署方式。
7. 起草 LoRA/QLoRA training plan 和 data schema。


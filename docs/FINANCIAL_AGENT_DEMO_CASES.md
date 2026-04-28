# Financial Agent Demo Cases

这组样本用于两类场景：

1. `GLM` vs `Financial Agent` 的并排展示
2. `Financial Agent` 是否真的调用了 `financial_analysis -> finma-sentiment-v2` 的功能验证

建议展示方式：

1. 在同一条输入下分别选择 `GLM` 和 `Financial Agent`
2. 重点观察 `Financial Agent` 是否先出现 `financial_analysis`
3. 对比最终回答在金融语义上的差异，而不是只看字数多少

## 推荐展示顺序

### 1. Mixed Sentiment

- `demo_003`
- 目标：展示 `beat + weak guidance` 这种混合信号时，Financial Agent 更不容易误判

### 2. Risk Labeling

- `demo_004`
- `demo_005`
- `demo_006`
- 目标：展示模型不只是说“偏负面”，而是能区分 `regulatory risk / operational risk / margin risk`

### 3. Management Tone

- `demo_010`
- `demo_011`
- `demo_012`
- 目标：展示管理层语气识别能力，尤其是 `cautiously optimistic` 和 `cautious`

### 4. Financial Signal Extraction

- `demo_013`
- `demo_014`
- `demo_015`
- 目标：展示模型能提取经营信号，而不是泛泛总结

## 使用建议

- 如果你们时间有限，优先演示 `demo_003 + demo_004 + demo_010 + demo_013`
- 如果你们要做海报或答辩 PPT，建议每类挑 1 个 case，做成 `输入 / GLM 输出 / Financial Agent 输出 / 观察点`
- 如果你们要录屏，先演示 `GLM`，再切到 `Financial Agent`，效果差异更容易被看见

## 数据文件

- 机器可读样本：[financial_agent_demo_cases.jsonl](/Users/yuk1no/6052/deer-flow/evals/financial_agent_demo_cases.jsonl)
- 当前说明文档：[FINANCIAL_AGENT_DEMO_CASES.md](/Users/yuk1no/6052/deer-flow/docs/FINANCIAL_AGENT_DEMO_CASES.md)

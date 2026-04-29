# Financial Agent Memory 简单测试方案

版本日期：2026-04-29  
适用分支状态：已接入 `Router + explicit memory retrieval + report skill memory_context + task memory write-back`

## 1. 测试目标

这份方案的目标不是证明“模型看起来更聪明了”，而是验证下面 4 件具体事情是否真的发生：

1. 系统能把有价值的信息写入长期任务记忆，而不是只停留在当前对话窗口里。
2. `context_memory_glm` 路由在追问场景下，能够复用前面已经写入的任务记忆。
3. `report_skill_glm` 路由在生成报告时，能够把历史结论和待办注入 `memory_context`。
4. `financial_finma` 短金融分析路线不会被项目类旧记忆污染。

## 2. 测试前准备

### 2.1 启动入口

浏览器访问：

```text
http://localhost:2026/
```

### 2.2 建议清空旧测试痕迹

为了避免历史数据干扰，建议在测试前备份或清空下面两个文件：

```text
backend/.deer-flow/memory_records.jsonl
backend/.deer-flow/memory.json
```

说明：

- `memory_records.jsonl` 是这次新加的“显式任务记忆”。
- `memory.json` 是 DeerFlow 原有的通用 memory。
- 如果不清空，也能测，但结果会混入旧上下文。

### 2.3 测试时重点观察的地方

1. 聊天回复最上方的 debug header
2. 最终回答是否承接上文
3. `backend/.deer-flow/memory_records.jsonl` 是否写入了结构化记录

你可以用下面命令观察显式任务记忆：

```bash
tail -n 20 backend/.deer-flow/memory_records.jsonl
```

## 3. 测试用例

---

## 用例 A：验证 Memory Write-back

### 目标

验证系统会把“用户明确要求记住的项目约束 / 主线”写入 `memory_records.jsonl`。

### 输入步骤

在同一个线程里输入：

```text
记住：这个项目后面的主线是 Router + Report Skill + Memory，后续实现都围绕这条线推进。先帮我简短总结一下这样做的原因。
```

### 预期现象

1. 回复能够正常返回。
2. 回复顶部的 debug header 一般会是：

```text
当前路由：general_glm 或 financial_glm
```

3. 执行：

```bash
tail -n 20 backend/.deer-flow/memory_records.jsonl
```

应当能看到至少 1 条新记录，常见类型包括：

- `user_requirement`
- `constraint`

### 通过标准

- 文件里出现了和“Router + Report Skill + Memory”主线相关的记录。
- 记录内容不是整段聊天原文，而是结构化摘要。

---

## 用例 B：验证 Context Follow-up 会复用 Memory

### 目标

验证 `context_memory_glm` 路由不仅识别“这是追问”，而且能承接前面已经保存的项目主线。

### 输入步骤

紧接着上一个用例，在同一个线程继续输入：

```text
把刚刚那个方案继续展开一下，重点说下一步要补什么。
```

### 预期现象

1. debug header 应显示：

```text
当前路由：context_memory_glm | memory=on
```

2. 回复中应该自然提到类似这些信息，而不是完全重新开题：

- Router
- Report Skill
- Memory
- 下一步补显式检索 / 写回 / 测试

### 通过标准

- 用户没有重新重复“Router + Report Skill + Memory”全称时，回答仍然能承接这个主线。
- 回答明显是在“继续前一个方案”，不是另起一段泛泛建议。

---

## 用例 C：验证 Report Skill 会吃到历史 Memory

### 目标

验证 `report_skill_glm` 生成报告时，会把前面的结论或待办带进去。

### 输入步骤

继续在同一个线程输入：

```text
针对刚刚的 memory 方案，给我生成一个简短报告。
```

### 预期现象

1. debug header 应显示：

```text
当前路由：report_skill_glm | memory=on | skill=on
```

2. 输出是 Markdown 报告。
3. 报告内容里应承接前面的项目语境，例如：

- 当前主线是 `Router + Report Skill + Memory`
- 下一步补显式 retrieval / write-back / evaluation
- 报告不是只围绕“memory”做空泛定义

### 通过标准

- 报告明显继承前文结论，而不是只根据最后一句“生成报告”临时发挥。
- 报告中不应该出现内部字段泄漏，例如：

```text
tool call
model_strategy
router_
system_reminder
```

---

## 用例 D：验证显式任务记忆真的写回

### 目标

验证报告生成后，系统会把摘要类记忆写回 `memory_records.jsonl`。

### 输入步骤

执行完用例 C 后，在终端查看：

```bash
tail -n 20 backend/.deer-flow/memory_records.jsonl
```

### 预期现象

应该看到新增记录，常见类型包括：

- `report_summary`
- `open_task`
- `project_fact`

### 通过标准

- 至少出现 1 条和刚生成报告相关的新任务记忆。
- 内容是摘要型，而不是把整篇报告全文保存进去。

---

## 用例 E：验证 FinMA 路线不会被项目记忆污染

### 目标

验证短金融分析路线默认不吃项目 memory，避免“你在聊财报，它却开始讲 Router/Skill/Memory”。

### 输入步骤

新开一个线程，输入：

```text
苹果公司第一季度财报增长 5%，这条信息偏积极还是偏消极？
```

### 预期现象

1. debug header 应显示：

```text
当前路由：financial_finma
```

或者是：

```text
当前路由：financial_glm
```

2. 回答应该只围绕财报、情绪、基本面、市场反应。
3. 回答里不应该出现下面这些无关项目词：

- Router
- Report Skill
- Memory
- 项目主线

### 通过标准

- 金融回答保持干净，没有被项目类历史记忆带偏。

---

## 用例 F：验证重复写入不会无上限膨胀

### 目标

验证同一条“记住”类输入重复出现时，不会把完全重复的记录无限追加。

### 输入步骤

在同一个线程里连续两次输入：

```text
记住：这个项目后面的主线是 Router + Report Skill + Memory。
```

### 预期现象

执行：

```bash
tail -n 50 backend/.deer-flow/memory_records.jsonl
```

你可能会看到不同类型的记录，但不应该每次都把完全相同的一条记录无穷复制。

### 通过标准

- 重复记录数量是可控的。
- 没有出现“同一类型 + 同一内容”被不断追加的明显膨胀。

## 4. 怎么判断我们的工作是有效的

如果你想要一个简单、答辩也能讲清楚的判定标准，可以直接用下面这组：

### 4.1 最低有效标准

满足下面 4 条，就可以认为这次工作“有效”：

1. `memory_records.jsonl` 能稳定写出结构化任务记忆。
2. `context_memory_glm` 的追问回答能承接前面已经确认的项目主线。
3. `report_skill_glm` 生成的报告能延续之前结论，而不是每次从零开始。
4. `financial_finma` 路由不会把无关项目记忆带进短金融分析。

### 4.2 更强的有效标准

如果下面这些也满足，就说明效果已经比较好：

1. 用户不需要反复重复背景信息。
2. 生成报告时会自然继承“下一步要做什么”“当前边界是什么”。
3. 重复信息不会让记忆文件无限膨胀。
4. 回答没有出现明显的无关历史污染。

## 5. 推荐给老师/答辩时的结论表达

如果测试通过，可以这样总结：

> 当前版本已经不再只是“识别这是一个多轮追问”，而是能把项目主线、约束、报告摘要和开放任务写入显式任务记忆，并在 `context_memory_glm` 与 `report_skill_glm` 路由中检索后再注入模型。这样系统在继续展开方案或生成报告时，能够更稳定地承接前面讨论过的内容，同时又避免把项目类旧记忆污染到短金融分析路由里。

## 6. 一句话验收结论

最实用的一句话标准是：

> 如果系统能“记住我们刚刚定下的项目主线”，并在后续追问和报告生成里继续沿着这条主线工作，同时又不会把这条记忆错误地带到苹果财报情绪分析里，那这次 memory 工作就算有效。

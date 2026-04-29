# Financial Agent RAG 技术文档

版本日期：2026-04-29  
适用仓库：`financial-agent`  
当前状态：已完成 `Router + Memory + Report Skill` 之后的 RAG V1 落地

---

## 1. 文档目的

本文档说明当前仓库中 RAG 模块的：

1. 设计目标与职责边界
2. 整体架构与数据流
3. 已实现的核心模块与代码落点
4. 与 Router、Memory、Report Skill 的集成方式
5. 当前版本边界
6. 推荐测试方案与评估指标

这份文档描述的是“当前真实实现”，不是一份脱离代码的理想化方案。

---

## 2. 设计目标

当前系统已经具备三层能力：

1. `Router`
   负责判断请求属于短金融文本分析、普通金融问答、上下文追问、通用问答还是报告生成。
2. `Memory`
   负责保存长期项目背景、阶段性结论、任务主线和开放任务。
3. `Report Skill`
   负责把报告生成拆成结构化流程，而不是只靠一次 prompt 直接写完。

RAG 的作用不是替代上面三层，而是补上第四层“外部证据层”：

1. 为普通问答提供可检索的项目文档或资料依据。
2. 为金融分析提供财报、公告、新闻、行业资料等证据上下文。
3. 为 Report Skill 提供 `retrieved_context`，让报告生成不只依赖模型常识。
4. 与 Memory 做清晰职责分离，避免“历史记忆”和“文档证据”混在一起。

一句话概括：

> Memory 负责“系统记得我们之前在做什么”，RAG 负责“系统能从资料里检索到什么证据”。

---

## 3. 职责边界

| 模块 | 存储内容 | 解决的问题 | 当前实现位置 |
|---|---|---|---|
| Conversation Context | 当前线程最近几轮对话 | 承接短期上下文 | LangGraph thread state |
| Memory | 项目背景、约束、历史结论、待办 | 支持多轮项目连续性 | `backend/packages/harness/deerflow/agents/memory/` |
| RAG | 本地文档 chunk 与检索证据 | 提供可引用的外部/项目内证据 | `backend/packages/harness/deerflow/agents/rag/` |
| Report Skill | 报告规划、写作、review、rewrite | 提升输出结构稳定性 | `backend/packages/harness/deerflow/agents/skills/` |

当前版本明确避免以下混淆：

1. 不把 RAG 检索结果直接写进长期 Memory。
2. 不把 Memory 事实当作文档证据引用。
3. 不把 RAG 设计成独立主 route，而是设计成 route 增强层。

---

## 4. 总体架构

当前实现采用“Router 判断是否开启 RAG + RAG Service 检索 + 不同路线消费证据”的结构。

```text
用户输入
  -> FinancialRoutingMiddleware
      -> route_decision
      -> memory_enabled / skill_enabled / rag_enabled
      -> rag_query / rag_source_type / rag_top_k
  -> RagService
      -> Query Rewrite
      -> Document Search
      -> Evidence Build
      -> RagBundle
  -> 按路线消费
      -> report_skill_glm: 注入 ReportSkillInput.rag_bundle / retrieved_context
      -> context_memory_glm: 作为 hidden evidence context 注入
      -> financial_glm: 作为 hidden evidence context 注入
      -> general_glm: 作为 hidden evidence context 注入
  -> LLM / Report Skill 输出
  -> Review / Citation Rule / Unsupported Claim Rule
```

当前设计中：

1. `financial_finma` 默认不启用 RAG。
2. `financial_glm`、`general_glm`、`context_memory_glm`、`report_skill_glm` 都可以开启 RAG。
3. `report_skill_glm` 是当前 RAG 使用最完整的一条路线。

---

## 5. 目录结构

### 5.1 RAG 模块

```text
backend/packages/harness/deerflow/agents/rag/
  __init__.py
  rag_schema.py
  document_loader.py
  text_splitter.py
  embedding_provider.py
  vector_store.py
  query_builder.py
  evidence_builder.py
  rag_service.py
```

### 5.2 构建脚本

```text
backend/scripts/build_rag_index.py
```

### 5.3 默认数据目录

```text
data/rag/raw/
data/rag/index/
```

其中：

1. `raw/` 放原始文档。
2. `index/` 存 `chunks.jsonl`、`vectors.json`、`metadata.json`。

---

## 6. 核心数据结构

当前实现定义了三类核心对象：

### 6.1 `RagChunk`

位置：`backend/packages/harness/deerflow/agents/rag/rag_schema.py`

用于表示离线入库后的文档块，包含：

1. `chunk_id`
2. `doc_id`
3. `source_path`
4. `title`
5. `section`
6. `text`
7. `chunk_index`
8. `source_type`
9. `metadata`

### 6.2 `RetrievedEvidence`

同样定义在 `rag_schema.py`，用于表示一次检索命中的证据块，额外包含：

1. `score`
2. `rank`

它支持两个关键输出格式：

1. `to_prompt_block()`
   生成 `[E1] / [E2]` 风格的 Evidence Context。
2. `to_context_record()`
   生成给 Report Skill 使用的结构化 evidence record。

### 6.3 `RagBundle`

用于表示一次完整检索结果，包含：

1. `query`
2. `rewritten_query`
3. `evidences`
4. `summary`
5. `missing`
6. `used`
7. `source_type`

当前设计里，`RagBundle` 是 Router、普通问答路线和 Report Skill 之间的统一交互对象。

---

## 7. 离线入库实现

### 7.1 文档加载

位置：`document_loader.py`

当前支持：

1. `.md`
2. `.txt`
3. `.pdf`

实现特点：

1. 目录输入会递归扫描支持的文件后缀。
2. Markdown 默认优先使用第一个标题作为 `title`。
3. PDF 通过 `pdfplumber` 抽取文本。
4. `source_type` 可以显式传入，也可以从目录名自动推断：
   `project_docs / finance_docs / eval_docs / user_upload`

### 7.2 文本切分

位置：`text_splitter.py`

当前切分策略是“标题感知 + 段落优先”的规则切分：

1. 先按 Markdown 标题切 section。
2. 再按段落聚合。
3. 超过阈值时按 `chunk_size` 和 `chunk_overlap` 做切片。

默认参数：

1. `chunk_size = 700`
2. `chunk_overlap = 100`

这是一个偏工程演示、稳定优先的 V1 策略。优点是实现简单、可复现、适合项目文档。缺点是对表格、复杂 PDF 版式和强结构化财务表处理还不够好。

### 7.3 向量化与索引

位置：`embedding_provider.py`、`vector_store.py`

当前版本没有接外部 embedding API，也没有接 FAISS，而是采用本地 `HashingEmbeddingProvider`：

1. 通过规则 tokenizer 抽取中英文 token。
2. 把 token 映射到固定维度哈希桶。
3. 对向量做归一化。

默认维度：

1. `dims = 256`

当前检索得分由三部分组成：

1. `cosine * 0.72`
2. `keyword_overlap * 0.22`
3. `title_bonus`

这意味着当前实现更准确地说是：

> 本地哈希向量检索 + 关键词重合度增强 + 标题命中加分

它不是“生产级语义向量库”，但非常适合 V1 原型、离线演示和课程/简历项目。

### 7.4 索引产物

构建后会在 `data/rag/index/` 生成：

1. `chunks.jsonl`
2. `vectors.json`
3. `metadata.json`

其中 `metadata.json` 至少包含：

1. `built_at`
2. `chunk_count`
3. `dims`

---

## 8. 在线检索实现

### 8.1 `RagService`

位置：`rag_service.py`

`RagService` 是当前 RAG 的统一入口，主要提供两个方法：

1. `search(...)`
2. `rebuild_index(...)`

`search(...)` 的流程：

1. 检查空查询。
2. 使用 `RagQueryBuilder` 重写 query。
3. 调用 `LocalVectorStore.search(...)` 执行检索。
4. 使用 `RagEvidenceBuilder` 把结果组装成 `RagBundle`。

### 8.2 Query Rewrite

位置：`query_builder.py`

当前 query rewrite 是轻量规则版，不依赖额外模型：

1. `report_skill_glm` 会把 `report_topic` 放进查询。
2. `context_memory_glm` 会拼接最近一段对话上下文。
3. 会从 `memory_context` 抽取少量关键词补到查询尾部。

这使得类似下面的追问不至于完全失去主题：

1. “那这个模块和 RAG 的区别呢？”
2. “把刚刚那个方案继续展开一下”
3. “针对上面的报告再写一版”

### 8.3 Evidence Build

位置：`evidence_builder.py`

当前 `RagEvidenceBuilder` 负责：

1. 在有命中时生成 `summary`
2. 在无命中时返回 `used=False`
3. 把命中结果包装成统一的 `RagBundle`

无命中时会显式写出：

1. `missing=["no_relevant_evidence"]`

这对后续 review 和“不足证据时必须说明限制”很重要。

---

## 9. Router 集成实现

位置：`backend/packages/harness/deerflow/agents/middlewares/financial_routing_middleware.py`

### 9.1 RouteDecision 扩展

当前 `RouteDecision` 新增了：

1. `rag_enabled`
2. `rag_query`
3. `rag_source_type`
4. `rag_top_k`
5. `rag_reason`

### 9.2 RAG 触发规则

当前实现是规则触发，不是分类模型触发。

触发依据包括：

1. 是否出现“文档 / 资料 / 证据 / 引用 / README / 年报 / 财报 / 新闻”等显式证据请求。
2. 当前 route 是否是 `report_skill_glm`。
3. 当前问题是否是金融问题。
4. 是否是项目文档问答。

### 9.3 source_type 推断

Router 会根据 query 规则推断：

1. `project_docs`
2. `finance_docs`
3. `eval_docs`
4. `auto`

例如：

1. 含 `router / report skill / memory / rag / 项目 / 文档 / 设计` 的问题，优先走 `project_docs`
2. 含 `财报 / 年报 / 公告 / 新闻 / company / industry / filing / earnings` 的问题，优先走 `finance_docs`
3. 含 `评估 / 实验 / eval / benchmark` 的问题，优先走 `eval_docs`

### 9.4 Hidden Evidence Context 注入

对于非 `report_skill_glm` 路线，Router 会把 `RagBundle.to_prompt_text()` 包成一个隐藏的 `system_reminder` 注入模型调用。

当前可以吃到这个 hidden evidence 的路线包括：

1. `general_glm`
2. `financial_glm`
3. `context_memory_glm`

`financial_finma` 默认不吃 RAG，避免短文本情绪分析被项目资料污染。

---

## 10. Report Skill 集成实现

位置：

1. `research_report_schema.py`
2. `research_report_skill.py`
3. `research_report_prompt.py`

### 10.1 `ReportSkillInput` 扩展

当前 `ReportSkillInput` 新增了：

1. `rag_enabled`
2. `rag_query`
3. `rag_source_type`
4. `rag_top_k`
5. `rag_bundle`
6. `require_citations`

### 10.2 检索接入点

当前 `ResearchReportSkill` 会在运行早期调用：

1. `_resolve_rag_bundle_sync(...)`
2. `_resolve_rag_bundle(...)`

然后通过：

1. `retrieve_evidence_sync(...)`
2. `retrieve_evidence(...)`

把 `RagBundle` 转成 `retrieved_context`，再交给写作流程。

### 10.3 Writer Prompt 规则

当 `require_citations=True` 且有 evidence 时，writer prompt 会显式要求：

1. 优先使用 Evidence Context 信息。
2. 不要编造证据中没有的年份、数字、排名和结论。
3. 关键结论尽量带 `[E1] [E2]` 样式引用。
4. 证据不足时要在风险或限制部分说明。

### 10.4 Review 规则

当前 `review_report(...)` 针对 RAG 增加了两类检查：

1. 引用覆盖检查
   如果 `require_citations=True` 且有 evidence，但报告中少于 2 个 `[E#]` 标记，会判定为需要 rewrite。
2. 证据不足说明检查
   如果 `rag_enabled=True` 但没有 evidence，报告必须明确说明“证据不足 / 需要进一步验证”等限制。

这使得当前版本不只是“会检索”，而是开始约束“如何使用检索结果”。

---

## 11. 构建脚本与使用方式

位置：`backend/scripts/build_rag_index.py`

### 11.1 默认用法

在仓库根目录执行：

```bash
backend/.venv/bin/python backend/scripts/build_rag_index.py
```

默认会读取：

```text
data/rag/raw/
```

### 11.2 指定输入

```bash
backend/.venv/bin/python backend/scripts/build_rag_index.py \
  --input /path/to/doc1.md \
  --input /path/to/folder \
  --source-type project_docs
```

### 11.3 当前仓库中的真实验证

当前实现已经在本地完成过一次真实构建，成功将多份项目文档切分并写入 `data/rag/index/`。这说明：

1. 构建脚本是可执行的。
2. 默认索引路径是可用的。
3. `RagService.search(...)` 可以基于真实文档返回命中结果。

---

## 12. 当前版本的真实能力边界

当前 RAG V1 已经实现：

1. 本地 `.md/.txt/.pdf` 文档入库
2. 规则式 query rewrite
3. 本地哈希向量检索
4. `RagBundle` 统一证据结构
5. Router 层 `rag_enabled` 增强
6. 普通 GLM 路线 hidden evidence 注入
7. Report Skill 证据消费与引用规则
8. 索引脚本与单元测试

当前 RAG V1 还没有实现：

1. 实时 Web 搜索
2. 外部 embedding API
3. FAISS / Milvus / pgvector 等正式向量库
4. BM25 或 Hybrid Search
5. Reranker
6. 复杂表格理解
7. 行级引用定位

因此当前版本更适合这样描述：

> 一个面向项目文档与本地金融资料的轻量级本地 RAG V1，实现了证据检索、路由增强和报告注入，但仍保留了进一步升级到真实 embedding / hybrid retrieval / rerank 的空间。

---

## 13. 当前已覆盖的测试

### 13.1 RAG Service 单测

位置：`backend/tests/test_rag_service.py`

当前已覆盖：

1. Markdown 文档可切分和入库
2. 查询 `Router` 能检索到对应文档
3. 无命中时能返回 `used=False`
4. Evidence Context 不泄漏 `chunk_id` 和 `score`

### 13.2 Router 集成测试

位置：`backend/tests/test_financial_routing_middleware.py`

当前已覆盖：

1. `report_skill_glm` 场景会打开 `rag_enabled`
2. 项目文档问题会切到 `project_docs`
3. 金融问题会切到 `finance_docs`
4. Router 会把 RAG evidence 作为 hidden context 注入

### 13.3 Report Skill 集成测试

位置：`backend/tests/test_research_report_skill.py`

当前已覆盖：

1. Report Skill 可以使用 `rag_bundle`
2. 报告可携带 `[E1] [E2]` 引用
3. 不泄漏内部字段
4. 证据不足时必须说明限制

---

## 14. 建议测试方案

为了让后续评估更系统，建议测试分三层做。

### 14.1 第一层：离线构建与检索测试

目标：确认入库和检索本身工作正常。

建议样例：

1. `Router 当前有哪些 route？`
2. `Report Skill 为什么比普通 prompt 更稳定？`
3. `Memory 和 RAG 的边界是什么？`
4. `基于项目文档解释 Router、Memory、Skill、RAG 的关系`

检查点：

1. `build_rag_index.py` 是否成功生成索引
2. `chunks.jsonl / vectors.json / metadata.json` 是否存在
3. `RagService.search(...)` 是否返回 `used=True`
4. Top-3 结果是否来自正确文档

### 14.2 第二层：Router 集成测试

目标：确认 RAG 会在合适的路线触发，而不是无差别打开。

建议样例：

1. `Router 文档里当前有哪些路线`
   预期：`general_glm + rag=on + source=project_docs`
2. `基于项目文档生成一份 Router + RAG 技术报告`
   预期：`report_skill_glm + rag=on + source=project_docs`
3. `苹果最新财报的主要风险是什么`
   预期：`financial_glm + rag=on + source=finance_docs`
4. `苹果财报增长8%`
   预期：`financial_finma + rag=off`
5. `把刚刚那个模块继续展开一下`
   预期：`context_memory_glm`，必要时结合 Memory 和 RAG

检查点：

1. Debug header 是否显示 `rag=on/off`
2. `rag_source=...` 是否合理
3. `financial_finma` 是否不会误吃 RAG

### 14.3 第三层：Report Skill 端到端测试

目标：确认报告生成真的使用了证据，而不是只是“路过检索”。

建议样例：

1. `根据项目文档生成一份 Router + Memory + RAG 技术说明`
2. `基于这些材料生成一份简短报告，并保留引用`
3. `如果当前资料不足，也请明确指出限制`

检查点：

1. 输出是否包含文档事实
2. 是否出现 `[E1] [E2]`
3. 是否避免编造不存在的年份或数字
4. 没有证据时是否明确说明限制
5. 是否没有泄漏 `chunk_id / score / rag_query`

### 14.4 第四层：回归测试

目标：确认引入 RAG 后没有破坏原有 Router + Memory + Skill 行为。

建议检查：

1. Memory 写回是否仍然正常
2. `context_memory_glm` 是否还能承接前文
3. `financial_finma` 是否仍保持短文本直接分析能力
4. UI 和服务启动流程是否不受影响

---

## 15. 建议评估指标

建议把指标分成“检索质量、回答质量、系统行为”三类。

### 15.1 检索质量指标

1. `Evidence Recall@K`
   含义：人工标注的 gold evidence 是否出现在 top-k 中。
2. `Context Relevance`
   含义：检索结果与问题是否相关。
3. `Source Type Accuracy`
   含义：`project_docs / finance_docs / eval_docs` 选择是否正确。

### 15.2 回答质量指标

1. `Faithfulness`
   含义：回答是否忠于证据。
2. `Unsupported Claim Rate`
   含义：出现无依据断言的比例。
3. `Citation Coverage`
   含义：关键结论中有多少带 `[E#]` 标记。
4. `Report Completeness`
   含义：报告是否覆盖要求章节。

### 15.3 系统行为指标

1. `RAG Trigger Precision`
   含义：该开 RAG 时开了，不该开时没乱开。
2. `Latency`
   含义：引入 RAG 后整体响应时间增长是否可接受。
3. `Fallback Robustness`
   含义：无证据时是否还能安全输出，不出现泄漏和幻觉扩张。

---

## 16. 推荐最小评测集

建议先做 20 到 30 条样例即可。

推荐分布：

1. 项目文档问答：8 条
2. 报告生成：8 条
3. 上下文追问：6 条
4. 金融资料问答：8 条

每条样例建议人工标注：

1. gold route
2. gold source_type
3. gold evidence 文档
4. 关键回答点
5. 是否需要 citation

这样后续就可以做小规模可复现实验。

---

## 17. 推荐结果展示表

可以使用如下表格展示实验结果：

| 方法 | Route Accuracy | Evidence Recall@5 | Faithfulness | Unsupported Claim Rate | Citation Coverage | Report Completeness |
|---|---:|---:|---:|---:|---:|---:|
| GLM Only | 待实验 | - | 待实验 | 待实验 | - | 待实验 |
| Router + Memory | 待实验 | - | 待实验 | 待实验 | - | 待实验 |
| Router + Memory + RAG | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| Full System | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

---

## 18. 后续优化建议

如果继续做 V2 / V3，建议优先顺序如下：

### 18.1 V2 优先项

1. 把 `HashingEmbeddingProvider` 升级成真实 embedding provider
2. 把 `vectors.json` 升级成正式向量库或 ANN 检索
3. 引入 BM25 或简单 hybrid search
4. 增加 `finance_docs` 样例语料

### 18.2 V3 优先项

1. 引入 reranker
2. 更细粒度的 citation check
3. 对财报表格和结构化 PDF 做专门处理
4. 增加 Web/API 检索缓存层

---

## 19. 简短结论

当前仓库中的 RAG 已经不是一个停留在设计图上的模块，而是一套可运行的 V1 实现：

1. 可以对本地文档做离线入库
2. 可以在 Router 层决定何时打开 RAG
3. 可以把证据注入普通问答路线
4. 可以把证据接入 Report Skill 写作与 review
5. 可以通过脚本、单测和真实文档索引进行验证

如果要对外概括这版工作，可以这样描述：

> 在原有 Router、Memory 与 Report Skill 基础上，实现了面向本地项目文档和金融资料的 RAG V1。该模块支持文档入库、标题感知切分、本地哈希向量检索、证据上下文注入，以及报告生成中的引用约束和证据不足回退逻辑，使系统从“基于对话与记忆生成”进一步扩展为“基于对话、记忆与外部证据协同生成”。

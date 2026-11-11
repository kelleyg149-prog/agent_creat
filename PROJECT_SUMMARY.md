# agent_creat 项目系统提示词

> 本文件是项目的「系统提示词 / 项目上下文」：任何新会话的 AI 接手本项目时，先完整阅读本文件，再执行任务。修改本项目架构后，须同步更新本文件。

---

## 1. 项目定位

基于 **LangGraph** 的 Agent 搭建学习项目。当前业务载体为「膜数据查询助手」：用户用自然语言提问，模型自主决定是否调用工具，工具调用须经人工审批，失败时依次走「LLM 自动修正 → 人工纠错 → 放弃说明」的容错链路。检索链路已升级为 **召回 → 重排 → 独立证据检查节点 → 生成/提示** 的 RAG 管线。

## 2. 技术栈与关键版本

| 组件 | 版本 / 说明 |
|---|---|
| Python | 3.14（项目自带 `.venv` 虚拟环境） |
| LangGraph | 1.2.15（`langgraph`、`langgraph-checkpoint` 4.3.0、`langgraph-prebuilt` 1.1.0） |
| LangChain | core 1.6.10、openai 1.7.1、huggingface 1.2.3、community 0.4.2（deepseek 1.1.1 为历史依赖） |
| 向量检索 | faiss-cpu 1.15.1 |
| Embedding | sentence-transformers 6.1.0（paraphrase-multilingual-MiniLM-L12-v2，384 维，CPU） |
| 推理模型 | **智谱 GLM-5.3**（OpenAI 兼容协议，base_url=`https://open.bigmodel.cn/api/paas/v4`；强制思考不可关：`thinking.type=enabled` + `reasoning_effort=low`） |
| 密钥 | `ZAI_API_KEY`（存于 `.env`，已被 `.gitignore` 排除，不入库） |

**换模型历史**：早期用 Moonshot Kimi K2.6，因**组织级 RPM=3 限流**（连续请求触发 429 崩溃）弃用，全量切换智谱。智谱 GLM-4-Flash 免费档 50 RPM 起、GLM-5.3 已官方确认存在（强制思考不可关）。

## 3. 模块结构与职责

| 文件 | 职责 | 状态 |
|---|---|---|
| `main.py` | 程序入口：持续对话循环；`safe_invoke()` 限流退避重试（429 后等 5s 重试 3 次）+ 认证/连接/格式/兜底异常友好提示；处理 `interrupt` 挂起的人工审批，循环 resume；**每轮结束调用 `extract_and_save_profile` 抽取长期记忆** | 工作区有未提交改动 |
| `config.py` | 配置中心：加载 `.env`、创建智谱 GLM-5.3 模型实例、`bind_tools`、`MAX_RETRY=3`、`MAX_CONTEXT_MESSAGES=40`、`COMPRESS_KEEP_RECENT=20`、**SQLite checkpointer（`data/checkpoint.sqlite`）**、**长期记忆 `SqliteStore`（`data/store.sqlite`，autocommit 连接 + `setup()` 建表）**、`thread_id` | 工作区有未提交改动 |
| `graph.py` | LangGraph 图：注册节点（llm / tool_node / check_node / auto_correct / retry / human_check / **evidence_check**）、条件边、`compile(checkpointer)`；`llm_node` 注入摘要模型、压缩保留条数与**长期记忆 store** | 工作区有未提交改动 |
| `nodes.py` | 全部节点实现（工厂函数模式）：llm（**超限摘要压缩 + 引入上下文注入长期记忆画像**）/ check_tool / auto_correct / retry / human_check / router / **create_evidence_check（RAG 答案透传：读子图产出的 answer 直接返回，写 State 证据字段）** | 同上 |
| `memory.py` | **记忆管理模块**：① `compress_context` 上下文压缩（超限 → `[历史摘要]`，RemoveMessage 删旧，ToolMessage 配对保护）② **长期记忆：`extract_and_save_profile`（LLM 按"稳定/可复用/不敏感"三标准抽取 → SqliteStore）+ `load_profile`（读取画像文本）** | 工作区有未提交改动 |
| `state.py` | `State(MessagesState)` 扩展字段（含 3 个证据字段）+ `init_state()` 统一初始化 | 同上 |
| `tools.py` | 工具定义（**6 个**）：`membrane_data`（三模式查询：编号/字段精确筛选/数值范围）、**`membrane_stats`（字段统计：count/max/min/mean）**、**`membrane_compare`（两膜字段对比：差值+更高者）**、**`membrane_top`（字段排名 Top-N，可升/降序）**、`calculater`、`rag_search`（内部调用 RAG 子图）；膜数据均从 `membrane.json` 实时读取 | 工作区有未提交改动 |
| `membrane.json` | 膜数据表（项目资产，**入库**；与 data/ 运行产物分开，后者 gitignored）；不存在时由 `_load_membrane()` 用内置默认首次生成 | 未入库（待 add） |
| `rag_engine.py` | RAG 检索基础能力：惰性加载 embedding/索引、FAISS 召回、LLM 重排、上下文组装；检索参数引用 `rag_config`；种子文档引用 `seed_documents`；**证据裁决与生成已上移子图，无 rag_search 入口** | 工作区有未提交改动 |
| `rag_config.py` | **RAG 检索参数集中配置（唯一出处）**：`RAG_CANDIDATE_K=5`、`RAG_RERANK_TOP_N=3`、`EVIDENCE_VECTOR_THRESHOLD=0.6`、`EVIDENCE_RERANK_THRESHOLD=5.0`；独立模块避免循环导入（config→tools→rag_engine） | 未入库 |
| `rag_subgraph.py` | **RAG 检索子图（SubGraph）**：retrieve(召回+重排) → evidence_check(双信号) → generate(LLM 生成) / no_evidence(友情提示)；输入 query、输出 answer；主图 `rag_search` 工具调用它，子图直接产出最终答案 | 未入库 |
| `seed_documents.py` | **种子文档共享模块**：`SEED_DOCUMENTS`（5 条）唯一出处，`rag-test.py` 与 `rag_engine.py` 共同引用，消除两份种子漂移 | 未入库 |
| `rag-test.py` | RAG 检索问答独立脚本（召回+重排+索引持久化+`--add` 增量；参数引用 `rag_config`，阈值 0.6 已与 engine 对齐） | 未入库 |
| `evaluate_rag.py` | 检索质量评估脚本：12 query（8 正例+2 弱相关+2 无关），Hit@1/Hit@3/Hit@5/MRR；参数引用 `rag_config` | 未入库 |
| `.env` | API Key（已 gitignore） | — |

## 4. 架构与执行流程

```
START → [llm]（消息 >40 条时先摘要压缩：旧消息 → [历史摘要]，RemoveMessage 删旧、保留最近 20 条）→ router
                 │
                 ├─ 无 tool_calls ─────────→ END（直接输出）
                 │
                 └─ 有 tool_calls ─→ [human_check] 人工审批（interrupt 挂起）
                                        │
                                        ├─ no 拒绝 → 补拒绝 ToolMessage → llm 收尾
                                        ├─ 「序号 新编号」改参 → 删旧 AIMessage 换新 → tool_node
                                        └─ yes 批准 → tool_node
                                                       │
                                                       ▼
                                                  [check_node] 校验结果
                                                       │
        ┌──────────────────────────────────────────────┼─────────────────────────────┐
  全部成功                                         失败且超限(≥3)                   失败未超限
  ↓                                                   → llm 自然语言说明收尾            │
  [evidence_check] RAG 答案透传（写 State 字段）                                        │
     │                                                ├─ 未 auto_corrected → [auto_correct] LLM 按错误信息修正参数
     ├─ 非 rag_search → llm 生成（常规工具）          │       ├─ 生成新 tool_calls → human_check（重新审批）
     └─ rag_search（子图已产出 answer）               │       └─ 仅文字说明 → llm
        直接把 answer 作为 AIMessage 返回 → END        └─ 已 auto_corrected → [retry] 人工纠错
                                                              ├─ yes 原样重试 / 「序号 新编号」改参 → tool_node
                                                              └─ no 放弃 → llm 说明失败原因

RAG 子图（rag_subgraph，由 rag_search 工具调用）：
  START → [retrieve](召回 k=5 + 重排 top_n=3)
              → [evidence_check] 双信号（向量 ≥0.6 且 重排分 ≥5.0）
                    ├─ 达标 → [generate] LLM 基于上下文生成答案 → answer → END
                    └─ 不足 → [no_evidence] 友情提示 → answer → END
```

### State 扩展字段语义

| 字段 | 类型 | 语义 |
|---|---|---|
| `tool_error` | bool | 本轮是否存在工具失败 |
| `retry_count` | int | 已重试次数（首次调用不计入；`MAX_RETRY=3`） |
| `tool_error_message` | str | 失败错误信息（供 auto_correct / retry 使用） |
| `human_approved` | bool | 最近一次工具调用是否获人工批准 |
| `auto_corrected` | bool | 是否已尝试过 LLM 自动修正 |
| `retrieval_success` | bool \| None | rag_search 是否执行成功（返回了候选）；None=本轮未执行检索或检索不适用 |
| `evidence_sufficient` | bool \| None | 证据双信号（向量+重排分）是否达标；None=未检查/不适用 |
| `evidence_reason` | str | 证据不足的原因说明（子图内生成，主图透传写入） |

## 5. 关键设计决策与约定

1. **工厂函数注入**：`nodes.py` 全部通过 `create_xxx(deps)` 工厂生成节点函数，依赖（模型、工具、常量）由 `graph.py` 注入，便于单元测试替换依赖。
2. **强制人工审批**：模型发起的任何 `tool_calls` 必须先经 `human_check` 的 `interrupt` 审批，模型没有自主执行工具的权限。
3. **三级失败处理**：工具失败 → ① LLM 自动修正一次（`auto_correct`）→ ② 人工纠错（`retry`，可原样重试或改膜编号）→ ③ 放弃并让 LLM 自然语言说明，程序不崩溃、可继续对话。
4. **状态清理防残留**：每轮链路结束（成功或超限）都清零 `retry_count / auto_corrected / tool_error*`；每轮 `main.py` 重置证据三字段，否则残留状态会让下一轮失败时跳过 auto_correct 直接判定超限，形成死循环。
5. **消息管理：压缩优先、裁剪兜底**：消息数 > `MAX_CONTEXT_MESSAGES=40` 时，`llm_node` 先走**摘要压缩**（`memory.compress_context`：早期消息 → LLM 摘要，保留最近 `COMPRESS_KEEP_RECENT=20` 条，用 `RemoveMessage` 删除旧消息使 State 持久精简，摘要只做一次）；压缩异常时回退 `trim_for_model` 硬裁剪（跳过开头 `ToolMessage`，避免破坏 tool_call 配对导致 API 400）。
6. **重试调用必须新 id**：原 tool_call 已执行完毕，重试时以 `call_retry_xxx` 新 id 追加，不能复用旧 id。
7. **模型配置**：智谱 GLM-5.3 强制思考（`extra_body` 传 `thinking.type=enabled` + `reasoning_effort=low`）；`timeout=60`（防代理 TLS 握手超时）+ `max_retries=3`（SDK 指数退避）。
8. **对话历史持久化（SQLite）**：checkpointer 由 `InMemorySaver` 升级为 `SqliteSaver`（依赖 `langgraph-checkpoint-sqlite` 包），历史写入 `data/checkpoint.sqlite`（已 gitignore），**跨进程记忆不丢**（实测：进程 1 记住名字，进程 2 仍能答出）；同一 `thread_id` 共享历史、换 `thread_id` 开新对话；连接需 `check_same_thread=False`（LangGraph 内部多线程访问 sqlite）；删除该文件即清空记忆。每轮 `graph.invoke` 只注入本轮 user 消息，并重置重试与证据相关字段。
9. **evidence_check 演进三步**：证据判断从 `rag_engine` 内部 → 主图独立节点（State 字段驱动：`retrieval_success` / `evidence_sufficient` / `evidence_reason`）→ **RAG 子图内节点**（`rag_subgraph.evidence_check_node`）。当前子图内完成 召回→重排→双信号裁决→生成/提示，主图 `evidence_check` 只透传子图 answer 并写 State 字段，不再重复裁决。
10. **多模式工具 docstring 驱动**：`membrane_data` 一个工具支持三种查询（按编号 / 按字段精确筛选 / 按数值范围），通过 docstring 写清用法示例，LLM 按问题自动选择模式与参数（实测三模式均正确调用）。
11. **safe_invoke 兜底**：`graph.invoke` 统一封装——限流（429）退避重试 3 次每次 5s，认证/连接/请求格式/兜底异常友好提示返回 None，程序永不因 API 异常崩溃。
12. **RAG 子图化（SubGraph）**：完整 RAG 管线封装为独立子图（`rag_subgraph.py`，输入 query 输出 answer），`rag_search` 工具调用子图、主图透传答案——主图保持"工具+审批+容错"骨架，检索细节完全封装；子图内证据不足直接产出提示，主图不再需要 no_evidence 节点（已移除）。重排分缺失（<0，重排回退）时退化为只依赖向量信号。
13. **上下文摘要压缩**：超限时早期消息 → LLM 摘要（`[历史摘要]` 前缀，保留用户身份/偏好、工具结论与关键数字、未完成待办），保留最近 20 条，`RemoveMessage` 从 State 删除旧消息（摘要只做一次）；tail 不以 ToolMessage 开头（配对保护）；失败回退硬裁剪。实测 88 条 → 删 68 条 + 20 条 + 1 条摘要，8 项断言全 PASS。
14. **三层记忆体系**：① 工作记忆 = State 临时字段（`retry_count` 等，每轮重置）② 会话记忆 = `SqliteSaver`（`checkpoint.sqlite`，按 thread_id 存全量消息，跨进程）③ **长期记忆 = `SqliteStore`（`store.sqlite`，namespace `user_profile`，跨 thread 不丢）**。长期记忆写入由 LLM 按"稳定 / 可复用 / 不敏感"三标准判定抽取（`extract_and_save_profile`，同 key 覆盖更新，失败静默跳过）；读取时由 `llm_node` 把画像拼成 SystemMessage 注入本次上下文——**不进 State**，避免每轮重复堆积。store 连接需 `isolation_level=None`（autocommit），否则 `setup()` 残留事务会让后续 `BEGIN` 报错。
15. **多工具并行调用**：模型一次返回多个 `tool_calls`（GLM-5.3 已实测支持），`human_check` 按序号列表一次审批全部，`ToolNode` 并行执行，`check_node` 收集连续 `ToolMessage` 统一校验。部分失败时走既有容错链：`auto_correct` 让模型判断（如编号不存在则放弃重试、直接说明），不会重复执行已成功的工具。审批改参限制：仅 `membrane_data` 支持「序号 新编号」，其他工具只支持 yes/no。

## 6. 当前进度（git 脉络）

已提交：`Initial commit` → gitignore → class state LangGraph → **模块化拆分**（`refactor/modularize` 分支）→ merge → 增加 rag 并接入 → **更换 api**（智谱）→ **增加兜底函数**（safe_invoke）。

未入库的工作区改动：
- `M`：`main.py`、`config.py`、`graph.py`、`nodes.py`、`memory.py`、`tools.py`、`evaluate_rag.py`、`rag-test.py`、`rag_engine.py`、`PROJECT_SUMMARY.md`（参数集中 + 膜数据外置 + 种子去重 + RAG 子图化 + SQLite 持久化 + 上下文压缩 + 长期记忆 + 多工具并行）
- `??`：`rag_config.py`（RAG 参数配置模块）、`rag_subgraph.py`（RAG 子图）、`membrane.json`（膜数据）、`seed_documents.py`（种子文档共享模块）

已完成能力：模块化 LangGraph 图、工具审批流（**支持多工具一次审批**）、**6 个业务工具（查询/统计/对比/排名/计算/检索）**、三级失败容错、上下文摘要压缩（超限压缩+失败回退裁剪）、SQLite 持久化会话记忆（跨进程不丢）、长期记忆（跨会话用户画像，三标准抽取+注入）、RAG 全链路（召回→重排→证据检查→生成/提示，已子图化）、检索质量评估、API 异常兜底、RAG 参数集中配置、膜数据外置、种子文档去重。

## 7. RAG 链路优化进度

### 已完成

1. **结构拆分**：检索链路提炼为独立引擎 `rag_engine.py`（召回 / 重排 / 证据检查剥离 / 上下文组装），惰性加载 embedding 模型与索引，主 Agent 启动不加载大模型；重排用独立无工具绑定 LLM 实例，防止工具内触发工具调用循环。
2. **数据外置**：`data/documents.json` 持久化文档元数据，首次用种子文档建库，之后以数据文件为准。
3. **索引持久化**：FAISS 索引 + 元数据落盘 `data/`，存在即加载、不存在才重建；`--add` 增量添加（`python rag-test.py --add "文本" --source x --page n --section s`）。
4. **重排（Reranker）**：`candidate_retrieval(k=5)` 粗召回 → LLM 相关性打分精排 Top-3（LLM-as-Reranker，本地无 cross-encoder 权重且离线无法下载；生产环境可替换 BGE-reranker 等）。已验证「静电排斥」从第 3 升到第 2。
5. **接入主 Agent**：`tools.py` 注册 `rag_search` 工具，复用既有审批流与三级失败容错链；已验证完整链路（提问 → 模型调用 rag_search → 人工审批 → 检索回答）。
6. **离线修复**：`HF_HUB_OFFLINE=1` 强制离线加载本地缓存，解决 huggingface.co 连接超时导致的加载失败。
7. **检索质量评估**：`evaluate_rag.py` 12 query（8 正例 + 2 弱相关 + 2 无关），Hit@1/MRR=0.833；重排回退时命中 0/12；发现无关 query「锂电池」在旧阈值 0.4 下误放行（向量 0.446），弱相关 pKa query 正确拦截（0.365<0.4）。
8. **evidence_check 独立节点**：证据裁决从 `rag_engine` 内部提升为 LangGraph 节点，双信号（向量 ≥0.6 且 重排分 ≥5.0）判断，State 字段驱动（`retrieval_success` / `evidence_sufficient` / `evidence_reason`），`no_evidence` 只读字段生成友情提示。三场景实测：正例走 LLM 生成、pKa 双信号拦截（0.53/4.0）走提示、membrane_data 放行。
9. **膜数据工具三模式**：`membrane_data` 支持按编号 / 按字段精确筛选（电压=3V）/ 按数值范围（膜通量>20），封装为三个查询函数 + 工具薄分发，docstring 驱动 LLM 选模式，三模式完整流程实测通过。
10. **参数集中配置**：新增 `rag_config.py` 作为 RAG 检索参数唯一出处（`RAG_CANDIDATE_K` / `RAG_RERANK_TOP_N` / `EVIDENCE_VECTOR_THRESHOLD` / `EVIDENCE_RERANK_THRESHOLD`），`rag_engine.py`、`nodes.py`（证据节点默认阈值）、`evaluate_rag.py`、`rag-test.py` 全部引用常量；独立模块避免循环导入（config→tools→rag_engine）；同时确认 `rag-test.py` 阈值已是 0.6，全部对齐。
11. **膜数据外置**：`membrane` 表从 `tools.py` 硬编码迁至项目根 `membrane.json`（业务资产，**入库**；`data/` 仍 gitignored 存运行产物）；`_load_membrane()` 每次查询实时读文件，改 JSON 立即生效（实测加 GO-4 不重启即可查询）；文件缺失时用内置默认首次生成，与 RAG 的 SEED 逻辑一致。
12. **种子文档去重**：`SEED_DOCUMENTS` 从 `rag-test.py` 与 `rag_engine.py` 两处抽出，统一到 `seed_documents.py` 共享模块（已验证两处引用同一对象，杜绝两份种子漂移）。
13. **RAG 子图化**：完整 RAG 管线封装为独立 SubGraph（`rag_subgraph.py`：retrieve → evidence_check → generate/no_evidence，输入 query 输出 answer），`rag_search` 工具调用子图，主图 `evidence_check` 透传 answer 并写 State 字段，主图 `no_evidence` 节点移除。子图独立单测：正例（向量 0.778/重排 10.0）生成答案、无关 query（向量 0.446/重排 0.0）双信号拦截出提示（顺带修复旧 0.4 阈值下的「锂电池」误放行）；主图三场景（正例直出答案 / pKa 提示 / GO-2 走 llm）实测通过。
14. **上下文摘要压缩**：新增 `memory.py`，消息超 `MAX_CONTEXT_MESSAGES=40` 时把早期消息压缩成 `[历史摘要]`（保留用户身份/偏好、工具结论与关键数字、未完成待办），保留最近 `COMPRESS_KEEP_RECENT=20` 条，`RemoveMessage` 从 State 删除旧消息 → 摘要只做一次、State 持续精简；tail 不以 ToolMessage 开头（配对保护）；压缩失败回退硬裁剪。单测 88 条 → 删 68 条 + 20 条 + 1 条摘要，8 项断言全 PASS；主图正常工具链路回归通过。
15. **长期记忆（跨会话用户画像）**：`SqliteStore`（`data/store.sqlite`，namespace `user_profile`）存用户稳定事实；写入 = `extract_and_save_profile`（LLM 按"稳定/可复用/不敏感"三标准抽取，同 key 覆盖，失败静默跳过），读取 = `llm_node` 注入画像 SystemMessage（不进 State）。实测：thread 1 说身份 → 抽取 `name / major / research_direction` 三条；**thread 2 全新会话**正确答出"金阿奎、资源与环境研究生、GO膜去除SMX"；负例"查一下GO-2"未写入；main.py 端到端正常。
16. **多工具并行调用 + 工具扩容**：新增 3 个膜数据工具（`membrane_stats` 统计 / `membrane_compare` 对比 / `membrane_top` 排名），工具总数 6；一句复合问题实测触发 3 个 `tool_calls` → 一次审批列出 `[1][2][3]` → 并行执行 → 汇总回答正确（均值 20.0、GO-2 高于 GO-1、GO-3 最高）；部分失败场景（GO-2 成功 + GO-99 失败）由 auto_correct 判定"编号不存在"直接自然语言说明，未重复执行成功工具。工具单测含错误用例（不存在字段/编号）全部通过。

### 待办

1. **证据阈值调优**：若后续要堵弱相关误放行，可再评估提高阈值或引入更多无关负例（0.6 下「锂电池」已正确拦截）。
2. **膜数据管理入口**：可仿照 `rag-test.py --add` 提供膜数据的命令行增删改入口（当前直接编辑 JSON 即可）。
3. **主图子图并行化**：后续可探索主图直接挂 RAG 子图节点（跳过工具调用层）或支持多工具并行调用。

## 8. 运行方式

```bash
# 对话式 Agent 入口（已接入 rag_search 工具 → 内部调用 RAG 子图）
.venv\Scripts\python.exe main.py

# RAG 单次检索问答（独立演示脚本，阈值与 engine 对齐 0.6）
.venv\Scripts\python.exe rag-test.py

# 增量添加文档到资料库（data/ 不存在时首次运行自动建库）
.venv\Scripts\python.exe rag-test.py --add "文档文本" --source 来源 --page 页码 --section 章节

# 检索质量评估（12 query 指标）
.venv\Scripts\python.exe evaluate_rag.py
```

退出指令：`end` / `END` / `结束`（大小写不敏感）。
审批指令：`yes` 批准（一次批准该组全部工具） / `no` 拒绝 / `「序号 新膜编号」` 修改参数后执行（如 `1 GO-2`；仅 `membrane_data` 支持，其他工具只支持 yes/no）。

## 9. 注意事项

- `.env` 含真实 API Key，**禁止**提交入库（已在 `.gitignore`）。
- `data/` 为运行产物（FAISS 索引 + 文档元数据 + **`checkpoint.sqlite` 对话历史** + **`store.sqlite` 长期记忆**），可重建/可清空，已在 `.gitignore` 排除；删除 `checkpoint.sqlite` 清空会话记忆（换 thread_id 即开新对话），删除 `store.sqlite` 清空长期用户画像（注意隐私：两文件都含对话/个人信息）。
- 本机无法直连 huggingface.co，embedding 模型加载依赖 `HF_HUB_OFFLINE=1` 使用本地缓存；重排使用 LLM-as-Reranker（本地无 cross-encoder 模型）。
- 智谱 API：GLM-5.3 强制思考不可关（`thinking.type=disabled` 会直接报错）；免费档有 RPM 限制（GLM-4-Flash 约 50 RPM），`safe_invoke` 已做 429 退避。
- 环境为 Windows；命令使用 PowerShell / `.venv\Scripts\python.exe`。
- 修改架构、新增模块、变更设计决策后，**必须同步更新本文件**。

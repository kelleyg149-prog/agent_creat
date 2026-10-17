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
| `main.py` | 程序入口：持续对话循环；`safe_invoke()` 限流退避重试（429 后等 5s 重试 3 次）+ 认证/连接/格式/兜底异常友好提示；处理 `interrupt` 挂起的人工审批，循环 resume | 已提交，工作区有未提交改动 |
| `config.py` | 配置中心：加载 `.env`、创建智谱 GLM-5.3 模型实例、`bind_tools`、`MAX_RETRY=3`、`MAX_CONTEXT_MESSAGES=40`、内存 checkpointer、`thread_id` | 已提交（更换 api 提交） |
| `graph.py` | LangGraph 图：注册节点（llm / tool_node / check_node / auto_correct / retry / human_check / **evidence_check / no_evidence**）、条件边、`compile(checkpointer)` | 工作区有未提交改动 |
| `nodes.py` | 全部节点实现（工厂函数模式）：llm / check_tool / auto_correct / retry / human_check / router / **create_evidence_check（证据双信号判断，写 State 字段）/ no_evidence（读 State 字段生成提示）** | 同上 |
| `state.py` | `State(MessagesState)` 扩展字段（含 3 个证据字段）+ `init_state()` 统一初始化 | 同上 |
| `tools.py` | 工具定义：`membrane_data`（**三模式查询**：按编号 / 按字段精确筛选 / 按数值范围）、`calculater`、`rag_search`（RAG 资料检索） | 同上 |
| `rag_engine.py` | RAG 检索引擎：FAISS 召回 → LLM 重排 → 返回结构化证据（含 `top_vector_score`/`top_rerank_score`/`retrieved_count`），**不再内置证据裁决**（交由图节点）；惰性加载 embedding/索引 | 工作区有未提交改动 |
| `rag-test.py` | RAG 检索问答独立脚本（召回+重排+索引持久化+`--add` 增量；**阈值仍是 0.4，未与 engine 的 0.6 对齐**） | 未入库 |
| `evaluate_rag.py` | 检索质量评估脚本：12 query（8 正例+2 弱相关+2 无关），Hit@1/Hit@3/Hit@5/MRR | 未入库（untracked） |
| `11.py` | hello world 极简脚本（学习/测试用） | 未入库 |
| `.env` | API Key（已 gitignore） | — |

## 4. 架构与执行流程

```
START → [llm] ─→ router
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
  [evidence_check] 证据检查（写 State 字段）                                             │
     │                                                ├─ 未 auto_corrected → [auto_correct] LLM 按错误信息修正参数
     ├─ 非 rag_search / 证据充足 → llm 生成            │       ├─ 生成新 tool_calls → human_check（重新审批）
     └─ 证据不足（双信号未达标）→ [no_evidence]        │       └─ 仅文字说明 → llm
        固定文案友情提示 → END                          └─ 已 auto_corrected → [retry] 人工纠错
                                                              ├─ yes 原样重试 / 「序号 新编号」改参 → tool_node
                                                              └─ no 放弃 → llm 说明失败原因
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
| `evidence_reason` | str | 证据不足的原因说明（列出未达标信号），供 no_evidence 生成提示 |

## 5. 关键设计决策与约定

1. **工厂函数注入**：`nodes.py` 全部通过 `create_xxx(deps)` 工厂生成节点函数，依赖（模型、工具、常量）由 `graph.py` 注入，便于单元测试替换依赖。
2. **强制人工审批**：模型发起的任何 `tool_calls` 必须先经 `human_check` 的 `interrupt` 审批，模型没有自主执行工具的权限。
3. **三级失败处理**：工具失败 → ① LLM 自动修正一次（`auto_correct`）→ ② 人工纠错（`retry`，可原样重试或改膜编号）→ ③ 放弃并让 LLM 自然语言说明，程序不崩溃、可继续对话。
4. **状态清理防残留**：每轮链路结束（成功或超限）都清零 `retry_count / auto_corrected / tool_error*`；每轮 `main.py` 重置证据三字段，否则残留状态会让下一轮失败时跳过 auto_correct 直接判定超限，形成死循环。
5. **消息裁剪保配对**：`MAX_CONTEXT_MESSAGES=40` 窗口裁剪时，跳过开头的 `ToolMessage`，避免裁掉它对应的 `AIMessage.tool_calls` 导致 API 400。
6. **重试调用必须新 id**：原 tool_call 已执行完毕，重试时以 `call_retry_xxx` 新 id 追加，不能复用旧 id。
7. **模型配置**：智谱 GLM-5.3 强制思考（`extra_body` 传 `thinking.type=enabled` + `reasoning_effort=low`）；`timeout=60`（防代理 TLS 握手超时）+ `max_retries=3`（SDK 指数退避）。
8. **对话历史持久化**：内存 checkpointer 按 `thread_id="1"` 跨轮次保留历史；每轮 `graph.invoke` 只注入本轮 user 消息，并重置重试与证据相关字段。
9. **evidence_check 独立节点（State 字段驱动）**：证据判断从 `rag_engine` 内部提升为图节点——`rag_search` 只做召回+重排并返回结构化证据分数，`evidence_check` 用**双信号**（向量相似度 ≥ 0.6 且 LLM 重排分 ≥ 5.0）裁决，结果写入 State 字段（`retrieval_success` / `evidence_sufficient` / `evidence_reason`），`no_evidence` 只读字段生成友情提示，两节点不重复解析工具结果。重排分缺失（<0，重排回退）时退化为只依赖向量信号。
10. **多模式工具 docstring 驱动**：`membrane_data` 一个工具支持三种查询（按编号 / 按字段精确筛选 / 按数值范围），通过 docstring 写清用法示例，LLM 按问题自动选择模式与参数（实测三模式均正确调用）。
11. **safe_invoke 兜底**：`graph.invoke` 统一封装——限流（429）退避重试 3 次每次 5s，认证/连接/请求格式/兜底异常友好提示返回 None，程序永不因 API 异常崩溃。

## 6. 当前进度（git 脉络）

已提交：`Initial commit` → gitignore → class state LangGraph → **模块化拆分**（`refactor/modularize` 分支）→ merge → 增加 rag 并接入 → **更换 api**（智谱）→ **增加兜底函数**（safe_invoke）。

未入库的工作区改动：
- `M`：`graph.py`、`main.py`、`nodes.py`、`rag_engine.py`、`state.py`、`tools.py`（evidence_check 节点 + State 字段 + 膜工具三模式）
- `??`：`evaluate_rag.py`（评估脚本，未提交）

已完成能力：模块化 LangGraph 图、工具审批流、三级失败容错、消息窗口裁剪、跨轮次记忆、RAG 全链路（召回→重排→证据检查节点→生成/提示）、检索质量评估、膜数据三模式查询、API 异常兜底。

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

### 待办

1. **参数集中配置**：`k / top_n / 证据阈值（0.6 / 5.0）/ 模型参数` 收进配置常量或环境变量（目前阈值散在 `graph.py` 的 `create_evidence_check()` 与 `evaluate_rag.py` 内联行）。
2. **阈值对齐**：`rag-test.py` 仍是 0.4，与 engine/评估的 0.6 未同步；若后续要堵「锂电池」类误放行，可再评估提高阈值或引入更多无关负例。
3. **种子数据去重**：`SEED_DOCUMENTS` 在 `rag-test.py` 与 `rag_engine.py` 各有一份，可抽为共享模块。
4. **膜数据外置**：`membrane` 表目前硬编码在 `tools.py`，可外置为 JSON/CSV 数据文件并支持增删改。

## 8. 运行方式

```bash
# 对话式 Agent 入口（已接入 rag_search 资料检索工具 + evidence_check 证据节点）
.venv\Scripts\python.exe main.py

# RAG 单次检索问答（独立演示脚本，阈值 0.4）
.venv\Scripts\python.exe rag-test.py

# 增量添加文档到资料库（data/ 不存在时首次运行自动建库）
.venv\Scripts\python.exe rag-test.py --add "文档文本" --source 来源 --page 页码 --section 章节

# 检索质量评估（12 query 指标）
.venv\Scripts\python.exe evaluate_rag.py
```

退出指令：`end` / `END` / `结束`（大小写不敏感）。
审批指令：`yes` 批准 / `no` 拒绝 / `「序号 新膜编号」` 修改参数后执行（如 `1 GO-2`）。

## 9. 注意事项

- `.env` 含真实 API Key，**禁止**提交入库（已在 `.gitignore`）。
- `data/` 为运行产物（FAISS 索引 + 文档元数据），可重建，已在 `.gitignore` 排除。
- 本机无法直连 huggingface.co，embedding 模型加载依赖 `HF_HUB_OFFLINE=1` 使用本地缓存；重排使用 LLM-as-Reranker（本地无 cross-encoder 模型）。
- 智谱 API：GLM-5.3 强制思考不可关（`thinking.type=disabled` 会直接报错）；免费档有 RPM 限制（GLM-4-Flash 约 50 RPM），`safe_invoke` 已做 429 退避。
- 环境为 Windows；命令使用 PowerShell / `.venv\Scripts\python.exe`。
- 修改架构、新增模块、变更设计决策后，**必须同步更新本文件**。

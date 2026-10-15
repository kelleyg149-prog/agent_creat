# agent_creat 项目系统提示词

> 本文件是项目的「系统提示词 / 项目上下文」：任何新会话的 AI 接手本项目时，先完整阅读本文件，再执行任务。修改本项目架构后，须同步更新本文件。

---

## 1. 项目定位

基于 **LangGraph** 的 Agent 搭建学习项目。当前业务载体为「膜数据查询助手」：用户用自然语言提问，模型自主决定是否调用工具，工具调用须经人工审批，失败时依次走「LLM 自动修正 → 人工纠错 → 放弃说明」的容错链路。

## 2. 技术栈与关键版本

| 组件 | 版本 / 说明 |
|---|---|
| Python | 3.14（项目自带 `.venv` 虚拟环境） |
| LangGraph | 1.2.15（`langgraph`、`langgraph-checkpoint` 4.3.0、`langgraph-prebuilt` 1.1.0） |
| LangChain | core 1.6.10、openai 1.7.1、huggingface 1.2.3、community 0.4.2、deepseek 1.1.1（历史依赖） |
| 向量检索 | faiss-cpu 1.15.1 |
| Embedding | sentence-transformers 6.1.0（paraphrase-multilingual-MiniLM-L12-v2，384 维，CPU） |
| 推理模型 | Moonshot Kimi K2.6（OpenAI 兼容协议，`thinking.type=disabled` 关闭思考模式） |
| 密钥 | `MOONSHOT_API_KEY`（存于 `.env`，已被 `.gitignore` 排除，不入库） |

## 3. 模块结构与职责

| 文件 | 职责 | 状态 |
|---|---|---|
| `main.py` | 程序入口：持续对话循环；处理 `interrupt` 挂起的人工审批，支持循环 resume | 已提交，工作区有未提交改动 |
| `config.py` | 配置中心：加载 `.env`、创建模型实例、`bind_tools`、`MAX_RETRY=3`、`MAX_CONTEXT_MESSAGES=40`、内存 checkpointer、`thread_id` | 同上 |
| `graph.py` | LangGraph 图：注册节点、条件边、`compile(checkpointer)` | 同上 |
| `nodes.py` | 全部节点实现（工厂函数模式）：llm / check_tool / auto_correct / retry / human_check / router | 同上 |
| `state.py` | `State(MessagesState)` 扩展字段 + `init_state()` 统一初始化 | 同上 |
| `tools.py` | 工具定义：`membrane_data`（膜数据查询）、`calculater`（两数相加）、`rag_search`（RAG 资料检索） | 已提交，新增 rag_search（未提交） |
| `rag_engine.py` | RAG 检索引擎：FAISS 召回 → LLM 重排 → 证据检查 → 上下文，惰性加载 embedding/索引，供 `rag_search` 工具调用 | **未入库（git untracked）** |
| `rag-test.py` | RAG 检索问答独立脚本（已优化：召回+重排+索引持久化+`--add` 增量） | 未入库 |
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
  清零 retry 状态 → llm                      → llm 自然语言说明收尾（不结束程序）      │
                                                                                      ├─ 未 auto_corrected → [auto_correct] LLM 按错误信息修正参数
                                                                                      │       ├─ 生成新 tool_calls → human_check（重新审批）
                                                                                      │       └─ 仅文字说明 → llm
                                                                                      └─ 已 auto_corrected → [retry] 人工纠错
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

## 5. 关键设计决策与约定

1. **工厂函数注入**：`nodes.py` 全部通过 `create_xxx(deps)` 工厂生成节点函数，依赖（模型、工具、常量）由 `graph.py` 注入，便于单元测试替换依赖。
2. **强制人工审批**：模型发起的任何 `tool_calls` 必须先经 `human_check` 的 `interrupt` 审批，模型没有自主执行工具的权限。
3. **三级失败处理**：工具失败 → ① LLM 自动修正一次（`auto_correct`）→ ② 人工纠错（`retry`，可原样重试或改膜编号）→ ③ 放弃并让 LLM 自然语言说明，程序不崩溃、可继续对话。
4. **状态清理防残留**：每轮链路结束（成功或超限）都清零 `retry_count / auto_corrected / tool_error*`，否则残留状态会让下一轮失败时跳过 auto_correct 直接判定超限，形成死循环。
5. **消息裁剪保配对**：`MAX_CONTEXT_MESSAGES=40` 窗口裁剪时，跳过开头的 `ToolMessage`，避免裁掉它对应的 `AIMessage.tool_calls` 导致 API 400。
6. **重试调用必须新 id**：原 tool_call 已执行完毕，重试时以 `call_retry_xxx` 新 id 追加，不能复用旧 id。
7. **模型配置**：Kimi K2.6 关闭思考模式（对齐早期 deepseek-chat 的直接问答行为）；`timeout=60`（走代理防 TLS 握手超时）+ `max_retries=3`（SDK 指数退避）。
8. **对话历史持久化**：内存 checkpointer 按 `thread_id="1"` 跨轮次保留历史；每轮 `graph.invoke` 只注入本轮 user 消息，并重置重试相关字段。

## 6. 当前进度（git 脉络）

已提交：`Initial commit` → gitignore 配置 → LangGraph state 类搭建 → **模块化拆分**（`refactor/modularize` 分支，规范 `graph.py` 命名）→ merge 回 main。

未入库的工作区改动：
- `M`（已修改未提交）：`.gitignore`（新增 `data/`）、`config.py`、`graph.py`、`main.py`、`nodes.py`、`state.py`、`tools.py`（新增 rag_search 工具）
- `??`（新增未跟踪）：`rag-test.py`、`rag_engine.py`、`PROJECT_SUMMARY.md`、`11.py`

已完成能力：模块化 LangGraph 图、工具审批流、三级失败容错、消息窗口裁剪、跨轮次记忆。**RAG 链路已打通并接入主 Agent**：`rag-search` 走既有审批流与容错链，资料库持久化在 `data/`（已 gitignore）。

## 7. RAG 链路优化进度

### 已完成

1. **结构拆分**：检索链路提炼为独立引擎 `rag_engine.py`（召回 / 重排 / 证据检查 / 上下文组装），惰性加载 embedding 模型与索引，主 Agent 启动不加载大模型；重排用独立无工具绑定 LLM 实例，防止工具内触发工具调用循环。
2. **数据外置**：`data/documents.json` 持久化文档元数据，首次用种子文档建库，之后以数据文件为准。
3. **索引持久化**：FAISS 索引 + 元数据落盘 `data/`，存在即加载、不存在才重建；`--add` 增量添加（`python rag-test.py --add "文本" --source x --page n --section s`）。
4. **重排（Reranker）**：`candidate_retrieval(k=5)` 粗召回 → LLM 相关性打分精排 Top-3（LLM-as-Reranker，本地无 cross-encoder 权重且离线无法下载；生产环境可替换 BGE-reranker 等）。已验证「静电排斥」从第 3 升到第 2。
5. **接入主 Agent**：`tools.py` 注册 `rag_search` 工具，复用既有审批流与三级失败容错链；已验证完整链路（提问 → 模型调用 rag_search → 人工审批 → 检索回答）。
6. **离线修复**：`HF_HUB_OFFLINE=1` 强制离线加载本地缓存，解决 huggingface.co 连接超时导致的加载失败。

### 待办

1. **检索质量评估**：构造多个 query + 标准答案，计算 hit rate / MRR / Top-k 命中，输出评估报告。
2. **参数集中配置**：`k / threshold / evidence_threshold / 模型参数` 收进配置常量或环境变量（目前 rag_engine 内硬编码 k=5、top_n=3、阈值 0.4）。
3. **种子数据去重**：`SEED_DOCUMENTS` 在 `rag-test.py` 与 `rag_engine.py` 各有一份，可抽为共享模块。

## 8. 运行方式

```bash
# 对话式 Agent 入口（已接入 rag_search 资料检索工具）
.venv\Scripts\python.exe main.py

# RAG 单次检索问答（独立演示脚本）
.venv\Scripts\python.exe rag-test.py

# 增量添加文档到资料库（data/ 不存在时首次运行自动建库）
.venv\Scripts\python.exe rag-test.py --add "文档文本" --source 来源 --page 页码 --section 章节
```

退出指令：`end` / `END` / `结束`（大小写不敏感）。
审批指令：`yes` 批准 / `no` 拒绝 / `「序号 新膜编号」` 修改参数后执行（如 `1 GO-2`）。

## 9. 注意事项

- `.env` 含真实 API Key，**禁止**提交入库（已在 `.gitignore`）。
- `data/` 为运行产物（FAISS 索引 + 文档元数据），可重建，已在 `.gitignore` 排除。
- 本机无法直连 huggingface.co，embedding 模型加载依赖 `HF_HUB_OFFLINE=1` 使用本地缓存；重排使用 LLM-as-Reranker（本地无 cross-encoder 模型）。
- 环境为 Windows；命令使用 PowerShell / `.venv\Scripts\python.exe`。
- 修改架构、新增模块、变更设计决策后，**必须同步更新本文件**。

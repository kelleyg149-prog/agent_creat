#RAG 检索参数集中配置（唯一出处）
#独立成模块的原因：依赖链 config → tools → rag_engine，若 rag_engine 反向 import config 会循环导入；
#rag_config 无任何依赖，rag_engine / nodes / evaluate_rag / rag-test 均可安全引用。

#粗召回数量（向量检索取前 N 个候选，交给重排精排）
RAG_CANDIDATE_K = 5

#重排后保留的前 N 条（最终进上下文的段落数）
RAG_RERANK_TOP_N = 3

#证据检查-向量相似度阈值（重排后首条候选的 FAISS 相似度下限）
EVIDENCE_VECTOR_THRESHOLD = 0.6

#证据检查-LLM 重排分阈值（模型对证据与问题相关性的打分，0-10）
EVIDENCE_RERANK_THRESHOLD = 5.0

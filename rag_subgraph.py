"""RAG 检索子图（SubGraph）：retrieve(召回+重排) → evidence_check(双信号) → generate / no_evidence。

主图通过 rag_search 工具调用本子图，子图直接产出最终答案（生成或友情提示），
主图的 evidence_check 节点只负责透传子图答案，不再重复做证据裁决。

依赖说明：
- 检索/重排复用 rag_engine 的惰性加载引擎与无工具绑定 LLM（_get_rerank_model），
  避免与主 Agent 的 tool-bound 模型混用（防工具内触发工具调用循环）；
- 本模块只依赖 rag_engine / rag_config，不 import config（config → tools → rag_engine 链上
  rag_engine 反向 import config 会循环导入，故子图也不触碰 config）。
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from rag_config import (
    RAG_CANDIDATE_K, RAG_RERANK_TOP_N,
    EVIDENCE_VECTOR_THRESHOLD, EVIDENCE_RERANK_THRESHOLD
)
from rag_engine import (
    get_engine, candidate_retrieval, rerank, _get_rerank_model, build_context
)


class RagState(TypedDict):
    query: str
    reranked: list              # 重排后的候选（中间态/输出用）
    vector_score: float         # 首条候选的 FAISS 向量相似度
    rerank_score: float         # 首条候选的 LLM 重排分（-1 = 重排回退）
    evidence_ok: bool           # 双信号是否达标
    evidence_reason: str        # 不足原因
    context: str                # 组装后的上下文（供 generate 使用）
    answer: str                 # 最终答案（generate 生成 或 no_evidence 提示）


#========== 节点 ==========

def retrieve_node(state: RagState) -> dict:
    """召回（FAISS Top-k）+ 重排（LLM 精排 Top-N），输出证据分数与上下文。"""
    embeddings, index, documents = get_engine()
    candidates = candidate_retrieval(state["query"], embeddings, index, documents, k=RAG_CANDIDATE_K)

    if not candidates:
        #资料库为空：视为证据不足，交给 no_evidence 提示
        return {
            "reranked": [],
            "vector_score": 0.0,
            "rerank_score": -1.0,
            "context": "",
            "evidence_ok": False,
            "evidence_reason": "资料库为空"
        }

    reranked = rerank(state["query"], candidates, _get_rerank_model(), top_n=RAG_RERANK_TOP_N)
    top = reranked[0] if reranked else {}
    return {
        "reranked": reranked,
        "vector_score": float(top.get("score", 0.0)),
        "rerank_score": float(top.get("rerank_score", -1.0)),
        "context": build_context(reranked),
    }


def evidence_check_node(state: RagState) -> Command:
    """证据双信号判断：向量相似度 + LLM 重排分都达标才生成答案，否则友情提示。"""
    vector_ok = state["vector_score"] >= EVIDENCE_VECTOR_THRESHOLD
    rerank_ok = state["rerank_score"] >= EVIDENCE_RERANK_THRESHOLD if state["rerank_score"] >= 0 else True
    passed = vector_ok and rerank_ok

    reasons = []
    if not vector_ok:
        reasons.append(f"语义相似度 {state['vector_score']:.2f}（阈值 {EVIDENCE_VECTOR_THRESHOLD}）")
    if not rerank_ok:
        reasons.append(f"相关性评分 {state['rerank_score']:.1f}/10（阈值 {EVIDENCE_RERANK_THRESHOLD}）")

    if passed:
        return Command(
            update={"evidence_ok": True, "evidence_reason": ""},
            goto="generate"
        )
    return Command(
        update={"evidence_ok": False, "evidence_reason": "、".join(reasons)},
        goto="no_evidence"
    )


def generate_node(state: RagState) -> dict:
    """基于检索上下文生成最终答案（LLM 只根据资料回答）。"""
    prompt = (
        "你是一名膜分离与新兴污染物研究助手。\n\n"
        "请只根据下面提供的参考资料回答问题。\n"
        '如果参考资料不足以回答，请明确说"现有资料不足以回答"。\n\n'
        f"参考资料：\n{state['context']}\n\n"
        f"用户问题：\n{state['query']}"
    )
    response = _get_rerank_model().invoke(prompt)
    return {"answer": response.content}


def no_evidence_node(state: RagState) -> dict:
    """证据不足：固定文案友情提示（不额外调用 LLM）。"""
    reason = state.get("evidence_reason") or "检索证据未达标"
    hint = (
        "抱歉，当前论文资料库中没有检索到足够相关的资料来回答这个问题"
        f"（{reason}）。\n"
        "建议换个问法，或先用 rag-test.py --add 补充相关资料后再试。"
    )
    return {"answer": hint}


#========== 编译子图 ==========

builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("evidence_check", evidence_check_node)
builder.add_node("generate", generate_node)
builder.add_node("no_evidence", no_evidence_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "evidence_check")
#evidence_check 通过 Command 自路由到 generate / no_evidence，不挂静态边

rag_subgraph = builder.compile()

"""RAG 检索质量评估：Hit Rate@k / MRR，对比「召回阶段」与「重排阶段」的排序质量。

评估集为人工构造的 query + 相关文档集合（ground truth 基于 data/documents.json 的索引顺序）。
用法：.venv\\Scripts\\python.exe evaluate_rag.py
"""
from rag_engine import (
    get_engine,
    candidate_retrieval,
    rerank,
    _get_rerank_model,
)


# 评估集：query -> 相关文档索引集合（0=pH去质子化 1=静电排斥 2=尺寸筛分 3=XPS 4=跨膜压力）
# 分三档：正例（语料明确覆盖）/ 弱相关（仅部分覆盖，测区分度）/ 无关（语料无内容，测误召回）
QUERIES = [
    # --- 正例：每篇文档都有明确对应问题 ---
    ("为什么提高pH可能增强SMX截留？", {0, 1}),
    ("GO膜如何影响污染物截留？", {1, 2}),
    ("膜孔径对截留有什么影响？", {2}),
    ("XPS能用来分析什么？", {3}),
    ("跨膜压力增大会怎样？", {4}),
    ("SMX在什么条件下容易去质子化？", {0}),
    ("膜表面通常带什么电荷？", {1}),
    ("如何表征膜表面的元素组成？", {3}),
    # --- 弱相关：语料仅部分覆盖，相关文档不应排第一 ---
    ("SMX的pKa具体数值是多少？", {0}),     # 文档0 只提去质子化，无 pKa 数值
    ("提高跨膜压力会增加什么？", {4}),       # 文档4 只涉及水通量
    # --- 无关：语料完全无相关内容，应触发证据检查失败 ---
    ("足球比赛越位规则是什么？", set()),
    ("如何给锂电池充电最安全？", set()),
]


def hit_rate(ranked_idx, relevant, k):
    """top-k 内是否至少命中一个相关文档（1/0）"""
    return 1 if any(idx in relevant for idx in ranked_idx[:k]) else 0


def mrr(ranked_idx, relevant):
    """第一个相关文档排名的倒数（未命中为 0）"""
    for rank, idx in enumerate(ranked_idx, 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def summarize(name, orders, relevants):
    """按排序集合汇总 Hit@1/3/5 与 MRR"""
    n = len(orders)
    return {
        "name": name,
        "hit@1": sum(hit_rate(o, r, 1) for o, r in zip(orders, relevants)) / n,
        "hit@3": sum(hit_rate(o, r, 3) for o, r in zip(orders, relevants)) / n,
        "hit@5": sum(hit_rate(o, r, 5) for o, r in zip(orders, relevants)) / n,
        "mrr": sum(mrr(o, r) for o, r in zip(orders, relevants)) / n,
    }


def main():
    print("===== RAG 检索质量评估 =====")
    embeddings, index, documents = get_engine()
    model = _get_rerank_model()
    print(f"语料文档数：{len(documents)} | 评估 query 数：{len(QUERIES)}\n")

    recall_orders, rerank_orders, relevants = [], [], []
    fallback_count = 0

    for qi, (query, relevant) in enumerate(QUERIES, 1):
        candidates = candidate_retrieval(query, embeddings, index, documents, k=5)
        recall_order = [c["idx"] for c in candidates]

        # 评估用全量重排（top_n=5 看完整排序）；主链路仍为 top_n=3
        reranked = rerank(query, candidates, model, top_n=5)
        rerank_order = [c["idx"] for c in reranked]

        if not any(c.get("rerank_score") is not None for c in reranked):
            fallback_count += 1

        recall_orders.append(recall_order)
        rerank_orders.append(rerank_order)
        relevants.append(relevant)

        # 逐 query 输出：两个阶段里相关文档的命中位置
        def rel_pos(order):
            pos = [str(i + 1) for i, idx in enumerate(order) if idx in relevant]
            return ", ".join(f"相关@#{p}" for p in pos) if pos else "未命中"

        # 证据检查模拟（重排后首条向量相似度 >= 0.6）：无关 query 应被拦截
        top_score = reranked[0]["score"] if reranked else 0.0
        evidence_pass = top_score >= 0.6

        print(f"[{qi}] {query}")
        print(f"     相关文档索引：{sorted(relevant)}")
        print(f"     召回阶段顺序：{[f'#{i+1}:{idx}' for i, idx in enumerate(recall_order)]}  → {rel_pos(recall_order)}")
        print(f"     重排阶段顺序：{[f'#{i+1}:{idx}' for i, idx in enumerate(rerank_order)]}  → {rel_pos(rerank_order)}")
        print(f"     证据检查：Top1 相似度 {top_score:.3f} → {'通过' if evidence_pass else '拦截（未检索到足够相关）'}")

    recall_s = summarize("召回阶段", recall_orders, relevants)
    rerank_s = summarize("重排阶段", rerank_orders, relevants)

    print(f"\n===== 汇总对比（{len(QUERIES)} 个 query 平均） =====")
    print(f"{'':10s} {'Hit@1':>8s} {'Hit@3':>8s} {'Hit@5':>8s} {'MRR':>8s}")
    for s in (recall_s, rerank_s):
        print(f"{s['name']:10s} {s['hit@1']:8.3f} {s['hit@3']:8.3f} {s['hit@5']:8.3f} {s['mrr']:8.3f}")

    print(f"\n重排回退次数：{fallback_count}/{len(QUERIES)}（回退=LLM 重排解析失败，按原始顺序）")


if __name__ == "__main__":
    main()

"""RAG 检索引擎：供主 Agent 的 rag_search 工具调用。

链路与 rag-test.py 一致（FAISS 召回 → LLM 重排 → 证据检查 → 上下文组装），
关键差异：
- 惰性加载：embedding 模型（约 470MB）与索引只在首次调用 rag_search 时初始化，
  避免主 Agent 一启动就加载大模型
- 重排用独立的无工具绑定 LLM 实例，防止工具内再次触发工具调用形成循环
- 返回结构统一为 {"success": bool, "data": str|None, "error": str|None}，与 tools.py 现有工具一致
"""
import faiss
import json
import numpy as np
import os

# 强制离线加载 embedding 模型（必须在 import langchain_huggingface 之前设置）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))

# 持久化路径（与 rag-test.py 共用同一份 data/ 索引，顺序即文档顺序）
DATA_DIR = os.path.join(_here, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")


# 种子文档：仅当 data/ 下还没有持久化索引时用于首次建库（与 rag-test.py 的种子一致）
SEED_DOCUMENTS = [
    {
        "text": "在较高pH条件下，SMX更容易发生去质子化，并表现出更强的负电性。",
        "source": "paper_01.pdf",
        "page": 8,
        "section": "3.3 Effect of pH"
    },
    {
        "text": "GO膜表面通常带负电，因此可能与阴离子型污染物产生静电排斥作用。",
        "source": "paper_01.pdf",
        "page": 9,
        "section": "3.4 Rejection mechanism"
    },
    {
        "text": "膜孔径和污染物分子尺寸会影响尺寸筛分作用。",
        "source": "paper_02.pdf",
        "page": 5,
        "section": "3.2 Size exclusion"
    },
    {
        "text": "XPS可以用于分析膜表面的元素组成和化学键状态。",
        "source": "paper_03.pdf",
        "page": 4,
        "section": "2.4 Characterization"
    },
    {
        "text": "提高跨膜压力通常会增加膜的水通量。",
        "source": "paper_04.pdf",
        "page": 7,
        "section": "3.1 Flux"
    }
]


# 惰性单例：模块导入时不加载任何大对象
_embeddings = None
_index = None
_documents = None
_rerank_model = None


def _get_rerank_model():
    """重排用 LLM：独立的无工具绑定实例，避免工具内再触发工具调用。"""
    global _rerank_model
    if _rerank_model is None:
        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            raise RuntimeError("未读取到 ZAI_API_KEY，请检查 .env 文件")
        _rerank_model = ChatOpenAI(
            model="glm-5.3",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            max_retries=3,
            timeout=60,
            extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        )
    return _rerank_model


def _save_index(documents, index):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, INDEX_PATH)


def _load_or_build_index(embeddings):
    """data/ 下已有索引与元数据则加载，否则用种子文档重建并落盘。"""
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        index = faiss.read_index(INDEX_PATH)
        return documents, index

    documents = list(SEED_DOCUMENTS)
    texts = [doc["text"] for doc in documents]
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    _save_index(documents, index)
    return documents, index


def get_engine():
    """惰性加载 embedding 模型与索引（首次调用时初始化）。"""
    global _embeddings, _index, _documents
    if _index is not None:
        return _embeddings, _index, _documents
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    _documents, _index = _load_or_build_index(_embeddings)
    return _embeddings, _index, _documents


def candidate_retrieval(query, embeddings, index, documents, k=5):
    """召回阶段（粗排）：FAISS 检索 Top-k 候选，不做阈值过滤（交给重排后处理）。"""
    query_vector = np.array([embeddings.embed_query(query)], dtype="float32")
    scores, indices = index.search(query_vector, k=k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "score": float(score),
            "document": documents[idx]
        })
    return candidates


def rerank(query, candidates, model, top_n=3):
    """精排阶段（Reranker）：LLM 按候选与问题的相关性打分排序（LLM-as-Reranker）。"""
    if len(candidates) <= 1:
        return candidates

    lines = "\n".join(
        f"[{i}] {c['document']['text']}" for i, c in enumerate(candidates)
    )
    rerank_prompt = (
        "你是检索重排器。判断下列每个候选片段与用户问题的相关性，"
        "输出 JSON 数组，按相关性从高到低排序。\n\n"
        f"用户问题：{query}\n\n"
        f"候选片段：\n{lines}\n\n"
        '输出格式（只输出 JSON，不要其他内容）：\n'
        '[{"idx": 0, "score": 0到10的整数, "reason": "一句话理由"}, ...]'
    )

    try:
        resp = model.invoke(rerank_prompt)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        scored = json.loads(text)
    except Exception:
        return candidates[:top_n]

    scored_map = {}
    for item in scored:
        try:
            scored_map[int(item["idx"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue

    if not scored_map:
        return candidates[:top_n]

    for i, c in enumerate(candidates):
        c["rerank_score"] = scored_map.get(i, -1.0)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]


def has_sufficient_evidence(retrieved_docs):
    """证据检查：重排后首条向量相似度不低于 0.4 才算检索质量合格。"""
    if not retrieved_docs:
        return False
    return retrieved_docs[0]["score"] >= 0.4


def build_context(retrieved_docs):
    """组装上下文文本：来源 + 页码 + 章节 + 正文（供模型阅读，不含调试分数）。"""
    context_parts = []
    for item in retrieved_docs:
        doc = item["document"]
        context_parts.append(
            f"来源：{doc['source']}（第{doc['page']}页，{doc['section']}）\n"
            f"内容：{doc['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def rag_search(query, top_n=3):
    """RAG 检索问答入口：召回 → 重排 → 证据检查 → 上下文，统一返回 success/data/error 结构。

    - 检索失败（资料库空、相似度过低、内部异常）返回 success=False 与原因，
      由主 Agent 的容错链路处理（自动修正 / 人工纠错 / 自然语言说明）。
    """
    try:
        embeddings, index, documents = get_engine()
        candidates = candidate_retrieval(query, embeddings, index, documents, k=5)

        if not candidates:
            return {
                "success": False,
                "data": None,
                "error": "资料库为空，未检索到任何候选文档"
            }

        reranked = rerank(query, candidates, _get_rerank_model(), top_n=top_n)

        if not has_sufficient_evidence(reranked):
            return {
                "success": False,
                "data": None,
                "error": "未检索到足够相关的资料（相似度过低），请换一种问法或先扩充资料库"
            }

        return {
            "success": True,
            "data": build_context(reranked),
            "error": None
        }
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "error": f"RAG 检索失败：{exc}"
        }

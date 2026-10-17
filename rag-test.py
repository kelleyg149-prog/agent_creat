import faiss
import json
import numpy as np
import os
import sys

# 强制离线加载 embedding 
# 模型已缓存在本地，离线模式可跳过联网检查直接使用缓存，避免启动时多次超时重试
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

api_key = os.getenv("ZAI_API_KEY")
if not api_key:
    raise RuntimeError("未读取到 ZAI_API_KEY，请检查 .env 文件是否存在且变量名正确")


# 持久化路径：FAISS 索引与文档元数据统一落在脚本同目录的 data/ 下
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")


# 种子文档：仅当 data/ 下还没有持久化索引时用于首次建库；
# 建库后数据以 data/documents.json 为准，增删文档直接改该文件或用 --add 增量添加
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


# 1. 初始化 Embedding 模型（多语言 MiniLM，384 维；normalize_embeddings=True 配合后续 IndexFlatIP 实现余弦相似度）
# 无论加载还是重建索引，query 编码都需要它
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# 2. 推理模型（重排序与最终回答共用；GLM-5.3 强制开启思考，采用轻量推理档位）
model = ChatOpenAI(
    model="glm-5.3",
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4",
    max_retries=3,
    timeout=60,
    extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
)


def save_index(documents, index):
    """把文档元数据与 FAISS 索引一起落盘。

    FAISS 索引只存向量，检索返回的序号与 index.add 顺序一一对应，
    因此文档元数据必须按同一顺序单独持久化，加载时才能对齐。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, INDEX_PATH)
    print(f"已保存索引：{INDEX_PATH}（{index.ntotal} 条向量）")


def load_or_build_index():
    """索引持久化入口：data/ 下已有索引与元数据则直接加载（跳过向量化），否则用种子文档重建并落盘。"""
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print(f"加载已有索引：{INDEX_PATH}")
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        index = faiss.read_index(INDEX_PATH)
        # 防御：索引向量数与元数据条数不一致说明数据文件被手动改坏
        if index.ntotal != len(documents):
            print(f"警告：索引向量数 {index.ntotal} 与元数据条数 {len(documents)} 不一致，"
                  "建议删除 data/ 目录后重新构建")
        return documents, index

    print("未找到持久化索引，用种子文档重建并保存…")
    documents = list(SEED_DOCUMENTS)
    texts = [doc["text"] for doc in documents]
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    save_index(documents, index)
    return documents, index


def add_documents(new_docs):
    """增量添加文档：编码新文档 → 追加进索引 → 元数据同步追加 → 重新落盘。"""
    documents, index = load_or_build_index()
    texts = [doc["text"] for doc in new_docs]
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    index.add(vectors)
    documents.extend(new_docs)
    save_index(documents, index)
    print(f"已增量添加 {len(new_docs)} 条，当前共 {len(documents)} 条文档")
    return documents, index


def candidate_retrieval(query, k=5):
    """召回阶段（粗排）：FAISS 向量检索 Top-k 候选。

    只做召回、不做阈值过滤——弱相关但有价值的候选交给 rerank 精排后再淘汰，
    避免在召回阶段误杀（阈值过滤已移到 rerank 之后）。
    """
    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector], dtype="float32")

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


def rerank(query, candidates, top_n=3):
    """精排阶段（Reranker）：对候选与问题的相关性重排序。

    标准 Reranker 是 cross-encoder（如 BGE-reranker），需单独模型权重；
    本机离线无法下载，故用 LLM 直接判断相关性（LLM-as-Reranker），可解释、零下载。
    返回按相关性降序的 top_n 候选；LLM 输出解析失败时回退为原向量分数顺序。
    """
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
        # 容错：剥掉可能的 ```json 代码块包裹
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        scored = json.loads(text)
    except Exception as exc:
        print(f"重排输出解析失败，回退原始顺序：{exc}")
        return candidates[:top_n]

    # 模型返回的 idx → 相关性分数 映射（解析异常的单条直接跳过）
    scored_map = {}
    for item in scored:
        try:
            scored_map[int(item["idx"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue

    if not scored_map:
        print("重排未返回有效分数，回退原始顺序")
        return candidates[:top_n]

    # 给每个候选挂上重排分（模型未覆盖的候选按 -1 垫底）
    for i, c in enumerate(candidates):
        c["rerank_score"] = scored_map.get(i, -1.0)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]


def has_sufficient_evidence(retrieved_docs):
    """证据检查：用重排后首条的向量相似度判断检索质量（0.6 为质量阈值）。"""
    if not retrieved_docs:
        return False

    best_score = retrieved_docs[0]["score"]

    if best_score < 0.6:
        return False

    return True


def build_context(retrieved_docs):
    """组装检索文档为上下文：向量相似度 + 重排分 + 元信息。"""
    context_parts = []

    for item in retrieved_docs:
        score = item["score"]
        rerank_score = item.get("rerank_score", "—")
        doc = item["document"]

        text = (
            f"向量相似度：{score:.3f}\n"
            f"重排分：{rerank_score}\n"
            f"来源：{doc['source']}\n"
            f"页码：{doc['page']}\n"
            f"章节：{doc['section']}\n"
            f"内容：{doc['text']}"
        )

        context_parts.append(text)

    return "\n\n---\n\n".join(context_parts)


# 3. 加载或重建持久化索引（data/ 下已有文件则跳过向量化直接加载）
documents, index = load_or_build_index()
print("索引向量数：", index.ntotal)


# 增量添加模式：python rag-test.py --add "文档文本" [--source 来源] [--page 页码] [--section 章节]
if "--add" in sys.argv:
    add_idx = sys.argv.index("--add")
    text = sys.argv[add_idx + 1] if len(sys.argv) > add_idx + 1 else ""
    if not text:
        print('用法：python rag-test.py --add "文档文本" [--source 来源] [--page 页码] [--section 章节]')
        sys.exit(1)
    new_doc = {
        "text": text,
        "source": sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "manual_add",
        "page": int(sys.argv[sys.argv.index("--page") + 1]) if "--page" in sys.argv else 0,
        "section": sys.argv[sys.argv.index("--section") + 1] if "--section" in sys.argv else "manual"
    }
    add_documents([new_doc])
    sys.exit(0)


# 4. 用户问题
query = "为什么提高pH可能增强SMX截留？"


# 5. 召回 Top-5 候选（FAISS 粗排，不做阈值过滤）
candidates = candidate_retrieval(query, k=5)
print("\n【召回阶段】FAISS Top-5：")
for i, c in enumerate(candidates, 1):
    print(f"  #{i}  向量 {c['score']:.4f}  {c['document']['text']}")

# 6. LLM 重排，精取 Top-3
reranked = rerank(query, candidates, top_n=3)
print("\n【重排阶段】Reranker Top-3：")
for i, c in enumerate(reranked, 1):
    print(f"  #{i}  重排分 {c.get('rerank_score', '—')}  向量 {c['score']:.4f}  {c['document']['text']}")

# 7. 证据检查：检索质量不足时直接终止
if not has_sufficient_evidence(reranked):
    print("警告：未检索到足够相关的文档（相似度阈值为 0.6）")
    raise SystemExit(1)

# 8. 组装上下文，调用 LLM 生成最终回答
context = build_context(reranked)

prompt = f"""
你是一名膜分离与新兴污染物研究助手。

请只根据下面提供的参考资料回答问题。
如果参考资料不足以回答，请明确说"现有资料不足以回答"。

参考资料：
{context}

用户问题：
{query}
"""

response = model.invoke(prompt)


print("\n===== RAG回答 =====")
print(response.content)

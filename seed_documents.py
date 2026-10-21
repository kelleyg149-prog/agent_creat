#种子文档：仅当 data/ 下还没有持久化索引时用于首次建库；
#建库后数据以 data/documents.json 为准，增删文档直接改该文件或用 --add 增量添加
#rag-test.py 与 rag_engine.py 共享此模块，避免两份种子漂移
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

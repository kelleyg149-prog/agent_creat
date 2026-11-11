#所有工具定义
import json
import os

from langchain_core.tools import tool

from rag_subgraph import rag_subgraph

#========== 膜数据加载（数据外置） ==========
#膜数据存于项目根 membrane.json（业务资产，入库；data/ 仍 gitignored 存运行产物）
#每次查询实时读文件：改 JSON 立即生效，无需重启 Agent
MEMBRANE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "membrane.json")

#内置默认数据：仅当 membrane.json 不存在时用于首次生成（与 RAG 的 SEED_DOCUMENTS 同理）
_DEFAULT_MEMBRANE = {
    "GO-1": {"膜通量": 10, "电压": 3.0, "离子通量": 56},
    "GO-2": {"膜通量": 20, "电压": 3.0, "离子通量": 56},
    "GO-3": {"膜通量": 30, "电压": 3.0, "离子通量": 56}
}


def _load_membrane() -> dict:
    """读取膜数据表；文件不存在时用内置默认生成一份，保证首次运行可用。"""
    if not os.path.exists(MEMBRANE_PATH):
        with open(MEMBRANE_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_MEMBRANE, f, ensure_ascii=False, indent=2)
    with open(MEMBRANE_PATH, encoding="utf-8") as f:
        return json.load(f)


#========== 膜数据查询逻辑封装 ==========

def _query_by_name(mem_name: str) -> dict:
    """按膜编号精确查询。"""
    membrane = _load_membrane()
    mem_name = mem_name.upper()
    if mem_name in membrane:
        return {"success": True, "data": membrane[mem_name], "error": None}
    return {"success": False, "data": None, "error": f"未找到编号 {mem_name}，可用编号：{', '.join(membrane)}"}


def _query_by_field(field: str, value: float) -> dict:
    """按数据字段精确筛选：返回该字段值等于 value 的所有膜。"""
    membrane = _load_membrane()
    matched = [
        {"膜编号": name, **data}
        for name, data in membrane.items()
        if data.get(field) == value
    ]
    if matched:
        return {"success": True, "data": {"count": len(matched), "membranes": matched}, "error": None}
    return {"success": False, "data": None, "error": f"没有膜的字段「{field}」等于 {value}"}


def _query_by_range(field: str, op: str, value: float) -> dict:
    """按数值范围筛选：op 支持 >、>=、<、<=、=。"""
    import operator
    membrane = _load_membrane()
    ops = {
        ">": operator.gt, ">=": operator.ge,
        "<": operator.lt, "<=": operator.le,
        "=": operator.eq
    }
    if op not in ops:
        return {"success": False, "data": None, "error": f"不支持的比较符 {op}，仅支持 >、>=、<、<=、="}
    matched = [
        {"膜编号": name, **data}
        for name, data in membrane.items()
        if isinstance(data.get(field), (int, float)) and ops[op](data[field], value)
    ]
    if matched:
        return {"success": True, "data": {"count": len(matched), "membranes": matched}, "error": None}
    return {"success": False, "data": None, "error": f"没有膜的字段「{field}」满足 {op} {value}"}


@tool
def membrane_data(
    mem_name: str | None = None,
    filter_field: str | None = None,
    filter_value: float | None = None,
    range_op: str | None = None,
    range_value: float | None = None,
):
    """查询膜实验数据，支持三种方式（每次只用其中一种）：

    1. 按膜编号查询单条数据：mem_name="GO-2"，返回该膜的 膜通量/电压/离子通量。
       可用膜编号：GO-1、GO-2、GO-3。

    2. 按字段精确筛选：filter_field + filter_value 找出字段值相等的所有膜，
       例如 filter_field="电压", filter_value=3.0 → 电压为 3V 的膜。
       可筛选字段：膜通量（数值）、电压（数值）、离子通量（数值）。

    3. 按数值范围筛选：filter_field + range_op + range_value 找出满足比较条件的所有膜，
       range_op 支持 >、>=、<、<=、=，
       例如 filter_field="膜通量", range_op=">", range_value=20 → 膜通量大于 20 的膜。

    返回数据中每条记录含 膜编号 与各字段值。
    """
    if mem_name:
        return _query_by_name(mem_name)
    if filter_field and filter_value is not None:
        return _query_by_field(filter_field, filter_value)
    if filter_field and range_op and range_value is not None:
        return _query_by_range(filter_field, range_op, range_value)
    return {"success": False, "data": None, "error": "查询条件不完整：请按三种方式之一提供参数（mem_name / filter_field+filter_value / filter_field+range_op+range_value）"}


@tool
def calculater(a: float, b: float):
    """将a和b相加"""
    return {
        "success": True,
        "data": a + b,
        "error": None
    }


@tool
def membrane_stats(field: str):
    """统计所有膜在某个字段上的总体数据（数量、最大值、最小值、平均值）。

    适用问题：膜通量的平均值是多少、离子通量最大是多少、电压有多少种取值等。
    参数 field 为字段名（膜通量、电压、离子通量）。
    """
    membrane = _load_membrane()
    values = [
        data[field] for data in membrane.values()
        if isinstance(data.get(field), (int, float))
    ]
    if not values:
        return {"success": False, "data": None, "error": f"字段「{field}」不存在或无数值，可用字段：膜通量、电压、离子通量"}
    return {
        "success": True,
        "data": {
            "field": field,
            "count": len(values),
            "max": max(values),
            "min": min(values),
            "mean": round(sum(values) / len(values), 4),
        },
        "error": None
    }


@tool
def membrane_compare(mem_name1: str, mem_name2: str, field: str):
    """对比两个膜在指定字段上的数值：返回两者数值、差值，并指出哪一方更高。

    适用问题：GO-1 和 GO-2 哪个膜通量高、GO-2 比 GO-3 的离子通量差多少。
    """
    membrane = _load_membrane()
    m1, m2 = mem_name1.upper(), mem_name2.upper()
    for name in (m1, m2):
        if name not in membrane:
            return {"success": False, "data": None, "error": f"未找到编号 {name}，可用编号：{', '.join(membrane)}"}
    v1, v2 = membrane[m1].get(field), membrane[m2].get(field)
    if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
        return {"success": False, "data": None, "error": f"字段「{field}」不存在或非数值，可用字段：膜通量、电压、离子通量"}
    return {
        "success": True,
        "data": {
            "field": field,
            f"{m1}": v1,
            f"{m2}": v2,
            "diff": round(v1 - v2, 4),
            "higher": m1 if v1 > v2 else (m2 if v2 > v1 else "相等"),
        },
        "error": None
    }


@tool
def membrane_top(field: str, n: int = 1, descending: bool = True):
    """按某字段排序，返回排名前 n 的膜（默认降序 = 数值最高的在前）。

    适用问题：膜通量最高的膜是哪个、离子通量最低的两个膜。
    n 为返回数量（默认 1），descending=False 时按升序（数值最低的在前）。
    """
    membrane = _load_membrane()
    rows = [
        {"膜编号": name, "值": data[field]}
        for name, data in membrane.items()
        if isinstance(data.get(field), (int, float))
    ]
    if not rows:
        return {"success": False, "data": None, "error": f"字段「{field}」不存在或无数值，可用字段：膜通量、电压、离子通量"}
    n = max(1, min(int(n), len(rows)))
    rows.sort(key=lambda r: r["值"], reverse=descending)
    return {
        "success": True,
        "data": {
            "field": field,
            "order": "降序" if descending else "升序",
            "top": rows[:n],
        },
        "error": None
    }


@tool
def rag_search(query: str):
    """在膜分离研究论文资料库中检索并回答与问题最相关的资料（含来源与页码）。

    适用于论文资料、机理机制、表征方法、研究结论等研究类问题，
    例如"为什么提高pH可能增强SMX截留？"。
    内部由 RAG 子图完成：召回 → 重排 → 证据检查 → 生成答案（或证据不足时给出提示）。
    """
    result = rag_subgraph.invoke({"query": query})
    return {
        "success": True,
        "data": {
            "answer": result["answer"],
            "top_vector_score": float(result["vector_score"]),
            "top_rerank_score": float(result["rerank_score"]),
            "evidence_ok": bool(result["evidence_ok"]),
            "evidence_reason": result["evidence_reason"],
            "retrieved_count": len(result["reranked"]),
        },
        "error": None
    }


tools = [membrane_data, membrane_stats, membrane_compare, membrane_top, calculater, rag_search]

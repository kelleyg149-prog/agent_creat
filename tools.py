#所有工具定义
from langchain_core.tools import tool

from rag_engine import rag_search as _rag_search

#膜实验数据表：字段为 膜通量(L·m⁻²·h⁻¹) / 电压(V) / 离子通量
membrane = {
    "GO-1": {
        "膜通量": 10,
        "电压": 3.0,
        "离子通量": 56
    },
    "GO-2": {
        "膜通量": 20,
        "电压": 3.0,
        "离子通量": 56
    },
    "GO-3": {
        "膜通量": 30,
        "电压": 3.0,
        "离子通量": 56
    }
}


#========== 膜数据查询逻辑封装 ==========

def _query_by_name(mem_name: str) -> dict:
    """按膜编号精确查询。"""
    mem_name = mem_name.upper()
    if mem_name in membrane:
        return {"success": True, "data": membrane[mem_name], "error": None}
    return {"success": False, "data": None, "error": f"未找到编号 {mem_name}，可用编号：{', '.join(membrane)}"}


def _query_by_field(field: str, value: float) -> dict:
    """按数据字段精确筛选：返回该字段值等于 value 的所有膜。"""
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
def rag_search(query: str):
    """在膜分离研究论文资料库中检索与问题最相关的段落（含来源与页码）。

    适用于论文资料、机理机制、表征方法、研究结论等研究类问题，
    例如"为什么提高pH可能增强SMX截留？"。返回按相关性精排后的 Top-3 段落。
    """
    return _rag_search(query)


tools = [membrane_data, calculater, rag_search]

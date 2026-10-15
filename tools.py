#所有工具定义
from langchain_core.tools import tool

from rag_engine import rag_search as _rag_search

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


@tool
def membrane_data(mem_name: str):
    """仅当用户给出膜编号时，给出膜编号对应的相关数据"""
    mem_name = mem_name.upper()
    if mem_name in membrane:
        return {
            "success": True,
            "data": membrane[mem_name],
            "error": None
        }
    else:
        return {
            "success": False,
            "data": None,
            "error": "未找到对应编号，请检查您的输入"
        }


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

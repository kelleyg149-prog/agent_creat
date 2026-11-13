#所有工具定义
import json
import os
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

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


#========== 多轮对话式检索（query 改写） ==========

_REWRITE_PROMPT = (
    "你是检索查询改写助手。根据下面的对话历史，把用户最新的检索问题改写成一个"
    "不依赖上下文、可独立检索的完整问题。\n"
    "规则：\n"
    "1) 若最新问题本身已完整（无「它/这个/那/其中」等指代），原样返回；\n"
    "2) 若存在指代，把对话历史中的主题（膜材料、污染物、机理名称等）补全进问题；\n"
    "3) 保留专业术语与关键实体，不要添加历史中没有的信息；\n"
    "4) 只输出改写后的问题本身，不要解释、不要加引号。"
)


def _recent_dialog(state, max_lines=8):
    """从 state.messages 提取最近的用户/助手文本（跳过 tool 与空内容）。"""
    if not state:
        return ""
    lines = []
    for msg in state.get("messages") or []:
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        if msg_type == "human" and content:
            lines.append(f"用户：{content}")
        elif msg_type == "ai" and content:
            lines.append(f"助手：{content}")
    return "\n".join(lines[-max_lines:])


def _rewrite_query(query, state):
    """多轮对话式检索：结合历史把追问改写为独立完整 query；无历史或失败时原样返回。"""
    history = _recent_dialog(state)
    if not history:
        return query
    try:
        from config import model  # 延迟导入：config → tools 存在依赖，运行时 config 已加载
        resp = model.invoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=f"对话历史：\n{history}\n\n最新检索问题：{query}\n\n改写结果："),
        ])
        rewritten = resp.content.strip() if isinstance(resp.content, str) else str(resp.content).strip()
        if not rewritten or len(rewritten) > 200:
            return query
        if rewritten != query:
            print(f"[查询改写] {query} → {rewritten}")
        return rewritten
    except Exception as exc:
        print(f"[查询改写失败，使用原 query] {exc}")
        return query


@tool
def rag_search(query: str, state: Annotated[dict, InjectedState]):
    """在膜分离研究论文资料库中检索并回答与问题最相关的资料（含来源与页码）。

    适用于论文资料、机理机制、表征方法、研究结论等研究类问题，
    例如"为什么提高pH可能增强SMX截留？"。
    支持多轮对话式检索：追问可以带指代（如"那静电排斥起什么作用？"），
    工具会结合对话历史把问题改写为独立完整的检索问题后再检索。
    内部由 RAG 子图完成：召回 → 重排 → 证据检查 → 生成答案（或证据不足时给出提示）。
    """
    rewritten = _rewrite_query(query, state)
    result = rag_subgraph.invoke({"query": rewritten})
    return {
        "success": True,
        "data": {
            "answer": result["answer"],
            "rewritten_query": rewritten,
            "top_vector_score": float(result["vector_score"]),
            "top_rerank_score": float(result["rerank_score"]),
            "evidence_ok": bool(result["evidence_ok"]),
            "evidence_reason": result["evidence_reason"],
            "retrieved_count": len(result["reranked"]),
        },
        "error": None
    }


tools = [membrane_data, membrane_stats, membrane_compare, membrane_top, calculater, rag_search]

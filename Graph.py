#LangGraph 流程：注册节点、边，编译图
from langgraph.graph import StateGraph, START, END
from config import checkpointer
from state import State
from nodes import llm_node, tool_node, check_tool, auto_correct_node, retry_node, human_check, router

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tool_node", tool_node)
builder.add_node("check_node", check_tool)
builder.add_node("auto_correct", auto_correct_node)
builder.add_node("retry", retry_node)
builder.add_node("human_check", human_check)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    router,
    {
        "human_check": "human_check",
        "end": END
    }
)

builder.add_edge("tool_node", "check_node")
#auto_correct 与 retry 都通过 Command 自行路由，不能挂静态边（会与 goto 重复路由）

graph = builder.compile(checkpointer=checkpointer)

#LangGraph 流程：注册节点、边，编译图
from langgraph.graph import StateGraph, START, END
from config import checkpointer, model_with_tool, MAX_RETRY, MAX_CONTEXT_MESSAGES
from tools import tools
from state import State
from nodes import (
    create_llm_node, create_auto_correct_node,
    create_check_tool, create_retry_node, create_tool_node,
    human_check, router
)

#通过工厂函数注入依赖，得到实际的节点函数
llm_node = create_llm_node(model_with_tool, MAX_CONTEXT_MESSAGES)
auto_correct_node = create_auto_correct_node(model_with_tool, MAX_CONTEXT_MESSAGES)
check_tool = create_check_tool(MAX_RETRY)
retry_node = create_retry_node()
tool_node = create_tool_node(tools)

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

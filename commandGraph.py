#模块加载
import json
from langgraph.types import Command
from typing import Literal
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()#加载。env文件

api_key = os.getenv("DEEPSEEK_API_KEY")
#执行openAi SDK
model=ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key
)

membrane = {
    "GO-1": {
        "膜通量": 25.6,
        "电压": 3.0,
        "离子通量": 56
    },
    "GO-2": {
        "膜通量": 25.6,
        "电压": 3.0,
        "离子通量": 56
    },
    "GO-3": {
        "膜通量": 25.6,
        "电压": 3.0,
        "离子通量": 56
    }
}

@tool
def membrane_data(mem_name:str):
    """仅当用户给出膜编号时，给出膜编号对应的相关数据"""
    mem_name=mem_name.upper()
    if mem_name in membrane:
        return {
            "success":True,
            "data":membrane[mem_name],
            "error":None
        }
    else:
        return {
            "success":False,
            "data":None,
            "error":"未找到对应编号，请检查您的输入"
        }


@tool
def calculater(a:float,b:float):
    """将a和b相加"""
    return  {
        "success": True,
        "data": a + b,
        "error": None
    }
tools=[membrane_data,calculater]


def check_tool(state) ->Command[Literal["retry","llm",]]:
    tool_answer=state["messages"][-1]
    result=json.loads(tool_answer.content)
    if result["success"]:
        return Command(
            update={
                "tool_error":False
            },
            goto="llm"
        )

    if state ["retry_count"]>=3:
        return Command(
            update={"tool_error":True},
            goto=END)

    return Command(
            update={
                "tool_error":True,
                "tool_error_message": result["error"]
                    },
            goto="retry")


#固定数据类型
class State(MessagesState):
    tool_error:bool
    retry_count:int
    tool_error_message:str
    #给模型绑定工具，而不是在每条请求时附加tool Schema
model_with_tool=model.bind_tools(tools)

#llm执行函数
def llm_node(state):
    response=model_with_tool.invoke(
        state["messages"]
    )
    return {
        "messages":[response]
    }

#工具调用def
tool_node=ToolNode(tools)

def router(state):
    assistant_message = state["messages"][-1]
    if not assistant_message.tool_calls:
        return "end"
    else:
        return "human_check"


def retry_node(state):
    return {
        "retry_count": state["retry_count"] + 1,
        "messages": [
            {
                "role": "user",
                "content": f"工具执行失败。\n错误原因：{state['tool_error_message']}\n请检查参数并重新调用工具。"
            }
        ]
    }

def human_check(state)-> Command[Literal["tool_node","END"]]:
    answer=interrupt("是否继续执行该工具")
    if answer=="yes":
        return Command(
            update={"human_approved":True},
            goto="tool_node")
    elif answer=="no":
        return Command(
            update={"human_approved":False},
            goto=END)
    return Command(
        update={"human_approved":False},
        goto=END)

def ask(state):
    answer = interrupt("请输入 yes 或 no")

    return {
        "answer": answer
    }

config = {
    "configurable": {
        "thread_id": "1"
    }
}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tool_node",tool_node)
builder.add_node("check_node",check_tool)
builder.add_node("retry",retry_node)
builder.add_edge(START, "llm")
builder.add_node("human_check",human_check)


builder.add_conditional_edges(
    "llm",
    router,
    {
        "human_check": "human_check",
        "end": END
    }
)

builder.add_edge("tool_node", "check_node")
builder.add_edge("retry","llm")

state:State = {
    "messages": [
        {
            "role": "user",
            "content":"查询go-1的数据"
        }],
        "tool_error":False,
        "retry_count":0
}

graph = builder.compile(checkpointer=checkpointer)

result = graph.invoke(state, config)

print(json.dumps(result,ensure_ascii=False,indent=4,default=str))
answer = input("请输入 yes 或 no：")

result = graph.invoke(
    Command(resume=answer),
    config
)

print(result)
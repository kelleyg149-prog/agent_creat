#所有节点：LLM、工具检查、自动修正、人工纠错、人工审批、路由
import json
import uuid
from typing import Literal

from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

from config import model, MAX_RETRY, MAX_CONTEXT_MESSAGES
from state import State
from tools import tools

#给模型绑定工具，而不是在每条请求时附加tool Schema
model_with_tool = model.bind_tools(tools)

#工具调用节点
tool_node = ToolNode(tools)


def trim_for_model(messages):
    """返回供模型使用的消息窗口：只保留最近的消息，且不破坏 tool_call 与 ToolMessage 的配对。"""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    window = messages[-MAX_CONTEXT_MESSAGES:]
    #窗口开头不能是 ToolMessage（它响应的 AIMessage 已被裁掉，会导致 API 400）
    i = 0
    while i < len(window) and window[i].type == "tool":
        i += 1
    return window[i:] or messages[-1:]


def check_tool(state) -> Command[Literal["retry", "llm", "auto_correct"]]:
    #收集本轮所有工具执行结果（消息末尾连续的 ToolMessage）
    tool_messages = []
    for msg in reversed(state["messages"]):
        if msg.type != "tool":
            break
        tool_messages.append(msg)

    errors = []
    for msg in tool_messages:
        result = json.loads(msg.content)
        if not result["success"]:
            errors.append(result["error"])

    if not errors:
        #本轮工具全部成功：清零本次重试相关状态，避免残留状态干扰后续链路
        return Command(
            update={
                "tool_error": False,
                "tool_error_message": "",
                "retry_count": 0,
                "auto_corrected": False
            },
            goto="llm"
        )

    if state["retry_count"] >= MAX_RETRY:
        #多次重试仍失败：不再结束程序，让模型直接用自然语言向用户说明，本轮正常收尾后可继续对话
        return Command(
            update={
                "tool_error": True,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"工具多次重试仍失败，错误信息：{'；'.join(errors)}。"
                        "请不要再调用工具，直接用自然语言向用户说明查询失败的原因，"
                        "并提示用户核对编号后重新输入。"
                    )
                }]
            },
            goto="llm")

    #失败且尚未自动修正过 → 先尝试 LLM 自动修正
    if not state["auto_corrected"]:
        return Command(
            update={
                "tool_error": True,
                "tool_error_message": "；".join(errors)
            },
            goto="auto_correct")

    #自动修正已试过，仍失败 → 交给人工纠错节点
    return Command(
        update={
            "tool_error": True,
            "tool_error_message": "；".join(errors)
        },
        goto="retry")


#llm执行函数
def llm_node(state):
    response = model_with_tool.invoke(
        trim_for_model(state["messages"])
    )
    return {
        "messages": [response]
    }


#自动修正节点：LLM 根据错误信息重新生成 tool_calls
def auto_correct_node(state) -> Command[Literal["human_check", "llm"]]:
    """
    首次失败后尝试让模型自动修正参数。
    给模型一个明确的系统提示，告诉它上一次的调用失败了，请重新生成正确的 tool_calls。
    注意：模型生成的修正调用仍然是"模型发起的工具请求"，必须先经 human_check 人工审批。
    """
    error_msg = state.get("tool_error_message", "未知错误")
    correction_prompt = (
        f"上一步的工具调用失败了，错误信息如下：\n"
        f"{error_msg}\n\n"
        f"请根据错误信息，修正参数后重新调用工具。"
    )
    #把错误提示追加到消息末尾，让模型重试
    messages = trim_for_model(state["messages"]) + [{"role": "user", "content": correction_prompt}]
    response = model_with_tool.invoke(messages)
    #重新发起了工具调用 → 先经人工审批；只返回了文字说明 → 交给 llm 节点收尾
    if getattr(response, "tool_calls", None):
        return Command(
            update={"auto_corrected": True, "messages": [response]},
            goto="human_check")
    return Command(
        update={"auto_corrected": True, "messages": [response]},
        goto="llm")


def router(state):
    assistant_message = state["messages"][-1]
    if not assistant_message.tool_calls:
        return "end"
    else:
        return "human_check"


#人工纠错节点：自动修正仍失败后，不再让模型自由决定，直接把错误信息和调用交给人工处理
def retry_node(state) -> Command[Literal["tool_node", "llm"]]:
    error_msg = state.get("tool_error_message", "未知错误")

    #找到最近一次（失败的）工具调用，作为人工纠错的基准
    last_ai = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai = msg
            break
    if last_ai is None:
        #理论上不会发生：check_tool 失败意味着存在已应答的 tool_calls
        return Command(
            update={"retry_count": state["retry_count"] + 1},
            goto="llm")
    tool_calls = last_ai.tool_calls

    #输入保护：非法输入时重新 interrupt，直到拿到有效输入
    hint = ""
    while True:
        payload = {
            "question": f"自动修正仍失败（{error_msg}）。输入 yes 原样重试 / 「序号 新膜编号」修改参数重试 / no 放弃",
            "tool_calls": tool_calls
        }
        if hint:
            payload["hint"] = hint
        answer = str(interrupt(payload)).strip()
        hint = ""

        if not answer:
            hint = "输入不能为空"
            continue

        low = answer.lower()
        if low == "yes":
            #原样重试：原调用已执行完毕，只能以新 id 追加一次新调用（参数不变）
            new_tool_calls = [
                {**tc, "id": f"call_retry_{uuid.uuid4().hex[:8]}"}
                for tc in tool_calls
            ]
            return Command(
                update={
                    "retry_count": state["retry_count"] + 1,
                    "messages": [AIMessage(content="", tool_calls=new_tool_calls)]
                },
                goto="tool_node")
        elif low == "no":
            #人工放弃：让模型直接向用户说明失败原因
            return Command(
                update={
                    "retry_count": state["retry_count"] + 1,
                    "messages": [{
                        "role": "user",
                        "content": "人工已放弃重试。请不要再调用工具，直接向用户说明查询失败的原因。"
                    }]
                },
                goto="llm")

        #解析「序号 新膜编号」格式
        parts = answer.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            hint = "无法识别的输入，请输入 yes / no / 或「序号 新膜编号」"
            continue

        idx = int(parts[0])
        new_mem_name = parts[1].strip()

        if idx < 1 or idx > len(tool_calls):
            hint = f"序号 {idx} 越界，有效范围 1-{len(tool_calls)}"
            continue

        #只有膜数据查询工具支持修改膜编号参数
        if tool_calls[idx - 1]["name"] != "membrane_data":
            hint = f"工具 {tool_calls[idx - 1]['name']} 不支持修改膜编号，请换个序号或直接 yes/no"
            continue

        if not new_mem_name:
            hint = "膜编号不能为空"
            continue

        #复制整组 tool_calls（新 id），只修改指定序号的 mem_name
        new_tool_calls = []
        for i, old_tc in enumerate(tool_calls):
            new_tc = {**old_tc, "id": f"call_retry_{uuid.uuid4().hex[:8]}"}
            if i == idx - 1:
                new_tc["args"] = {**old_tc["args"], "mem_name": new_mem_name}
            new_tool_calls.append(new_tc)
        return Command(
            update={
                "retry_count": state["retry_count"] + 1,
                "messages": [AIMessage(content="", tool_calls=new_tool_calls)]
            },
            goto="tool_node")


def human_check(state) -> Command[Literal["tool_node", "llm"]]:
    tool_calls = state["messages"][-1].tool_calls

    #输入保护：非法输入时重新 interrupt，直到拿到有效输入
    hint = ""
    while True:
        payload = {
            "question": "输入 yes 批准 / no 拒绝 / 或「序号 新膜编号」修改参数，如：1 GO-2",
            "tool_calls": tool_calls
        }
        if hint:
            payload["hint"] = hint
        answer = str(interrupt(payload)).strip()
        hint = ""

        #空输入
        if not answer:
            hint = "输入不能为空"
            continue

        low = answer.lower()
        if low == "yes":
            #原样通过，执行原有工具调用
            return Command(
                update={"human_approved": True},
                goto="tool_node")
        elif low == "no":
            #拒绝执行：为每个 tool_call 补一条拒绝结果的 ToolMessage，
            #让模型能自然回应（否则消息链以未应答的 tool_calls 结尾会报 400）
            return Command(
                update={
                    "human_approved": False,
                    "messages": [
                        ToolMessage(
                            tool_call_id=tc["id"],
                            name=tc["name"],
                            content="用户拒绝了该工具调用"
                        )
                        for tc in tool_calls
                    ]
                },
                goto="llm")

        #解析「序号 新膜编号」格式
        parts = answer.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            hint = "无法识别的输入，请输入 yes / no / 或「序号 新膜编号」"
            continue

        idx = int(parts[0])
        new_mem_name = parts[1].strip()

        #序号越界
        if idx < 1 or idx > len(tool_calls):
            hint = f"序号 {idx} 越界，有效范围 1-{len(tool_calls)}"
            continue

        #只有膜数据查询工具支持修改膜编号参数，避免给其他工具塞进无效参数
        if tool_calls[idx - 1]["name"] != "membrane_data":
            hint = f"工具 {tool_calls[idx - 1]['name']} 不支持修改膜编号，请换个序号或直接 yes/no"
            continue

        #膜编号为空
        if not new_mem_name:
            hint = "膜编号不能为空"
            continue

        #复制整组 tool_calls，只修改指定序号的 mem_name，其余原样保留
        new_tool_calls = []
        for i, old_tc in enumerate(tool_calls):
            new_tc = {**old_tc}
            if i == idx - 1:
                new_tc["args"] = {**old_tc["args"], "mem_name": new_mem_name}
            new_tool_calls.append(new_tc)
        new_ai_message = AIMessage(
            content="",
            tool_calls=new_tool_calls
        )
        #先删除原 AIMessage（避免遗留未应答的 tool_calls），再追加新的
        return Command(
            update={
                "human_approved": True,
                "messages": [
                    RemoveMessage(id=state["messages"][-1].id),
                    new_ai_message
                ]
            },
            goto="tool_node")

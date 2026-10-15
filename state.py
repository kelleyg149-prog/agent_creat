#State 定义
from typing import Any

from langgraph.graph import MessagesState


class State(MessagesState):
    tool_error: bool
    retry_count: int
    tool_error_message: str
    human_approved: bool
    auto_corrected: bool


#各字段的初始值（集中管理，避免散落在调用方）
_STATE_DEFAULTS = {
    "tool_error": False,
    "retry_count": 0,
    "tool_error_message": "",
    "human_approved": False,
    "auto_corrected": False
}


def init_state(user_input: str | None = None, **overrides: Any) -> dict:
    """统一的 State 初始化入口。

    - user_input：本轮用户输入，会作为第一条 user 消息写入 messages
    - overrides：按需覆盖任意字段的初始值（例如测试时注入特定 retry_count）
    """
    state = dict(_STATE_DEFAULTS)
    state["messages"] = []
    if user_input:
        state["messages"].append({"role": "user", "content": user_input})
    state.update(overrides)
    return state

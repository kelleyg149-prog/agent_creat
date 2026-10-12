#State 定义
from langgraph.graph import MessagesState


class State(MessagesState):
    tool_error: bool
    retry_count: int
    tool_error_message: str
    human_approved: bool
    auto_corrected: bool

"""上下文管理：超限时把早期消息用 LLM 压缩成摘要，替代硬裁剪（丢弃最旧消息）。

策略：
- 消息数 ≤ max_messages：不处理（零开销）
- 超过：保留最近 keep_recent 条（tail），更早的消息（head）交给 LLM 压缩成一段摘要，
  摘要以 SystemMessage("[历史摘要] ...") 放入消息开头，并用 RemoveMessage 从 State 删除旧消息
  → 摘要只做一次，State 持续精简，避免每轮重复总结
- 配对保护：tail 开头不能是 ToolMessage（它响应的 AIMessage 可能被划进 head，导致 API 400）
- 压缩失败抛异常，由调用方回退到硬裁剪（trim_for_model），保证对话不中断
"""
from langchain_core.messages import SystemMessage, HumanMessage

SUMMARY_PREFIX = "[历史摘要] "

_COMPRESS_PROMPT = (
    "把下面这段早期对话压缩成一段简洁的中文摘要，务必保留：\n"
    "1) 用户的身份信息、称呼、偏好、正在研究的方向；\n"
    "2) 工具查询得到的核心结论、关键数字、膜编号等事实；\n"
    "3) 尚未完成的待办或用户提出的后续要求。\n"
    "不要臆造内容，不要评论，只输出摘要正文。"
)

_ROLE_NAMES = {"human": "用户", "ai": "助手", "tool": "工具结果", "system": "系统"}


def _format_messages(messages):
    """把消息列表拍平成纯文本，供摘要模型阅读。"""
    lines = []
    for msg in messages:
        role = _ROLE_NAMES.get(msg.type, msg.type)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            content = f"[调用工具] {content}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def compress_context(messages, max_messages, keep_recent, model):
    """消息超限时压缩早期消息。

    返回 (removed_messages, summary_message) —— removed 需从 State 删除；
    未超限返回 None；压缩失败抛异常（调用方回退硬裁剪）。
    """
    if len(messages) <= max_messages:
        return None

    tail_start = max(len(messages) - keep_recent, 0)
    # 配对保护：tail 不能以 ToolMessage 开头（其 AIMessage 若被划进 head 会破坏配对）
    while tail_start < len(messages) and messages[tail_start].type == "tool":
        tail_start += 1
    head = list(messages[:tail_start])
    if not head:
        return None

    text = _format_messages(head)
    resp = model.invoke([
        SystemMessage(content=_COMPRESS_PROMPT),
        HumanMessage(content=text),
    ])
    summary = resp.content if isinstance(resp.content, str) else str(resp.content)
    return head, SystemMessage(content=SUMMARY_PREFIX + summary.strip())

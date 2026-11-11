"""记忆管理：① 会话内上下文压缩（超限摘要）② 跨会话长期记忆（用户画像）。

① 上下文压缩：
- 消息数 ≤ max_messages：不处理（零开销）
- 超过：保留最近 keep_recent 条（tail），更早的消息（head）交给 LLM 压缩成一段摘要，
  摘要以 SystemMessage("[历史摘要] ...") 放入消息开头，并用 RemoveMessage 从 State 删除旧消息
  → 摘要只做一次，State 持续精简，避免每轮重复总结
- 配对保护：tail 开头不能是 ToolMessage（它响应的 AIMessage 可能被划进 head，导致 API 400）
- 压缩失败抛异常，由调用方回退到硬裁剪（trim_for_model），保证对话不中断

② 长期记忆：
- 写入：每轮结束后 LLM 按"稳定/可复用/不敏感"三标准判定，抽取用户事实写入 SqliteStore
- 读取：每轮送给模型前把画像作为 SystemMessage 注入（不写进会话历史，避免每轮重复堆积）
"""
import json

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


# ===== 长期记忆（跨会话用户画像）=====

PROFILE_NAMESPACE = ("user_profile",)

_EXTRACT_PROMPT = """从本轮对话中判断哪些信息值得写入"用户长期记忆"（跨会话复用）。

只保留同时满足以下三条的信息：
1) 稳定：跨时间基本不变（姓名/称呼、职业或身份、研究方向、长期偏好、所在机构或城市）
2) 可复用：后续对话大概率会用到
3) 不敏感：不含密码、身份证号、银行卡号、手机号等隐私

一次性内容（本次查询的具体膜编号、临时计算数字）、闲聊情绪、寒暄不保存。

输出严格 JSON（不要任何其他文字）：
{"items": [{"key": "name", "value": "金阿奎"}], "reason": "简要说明"}
没有值得保存的信息时返回 {"items": [], "reason": "简要说明"}
"""


def load_profile(store):
    """读取全部长期记忆条目，拼成文本供模型阅读；无条目返回空字符串。"""
    if store is None:
        return ""
    items = store.search(PROFILE_NAMESPACE)
    if not items:
        return ""
    lines = []
    for item in items:
        value = item.value.get("value", "") if isinstance(item.value, dict) else str(item.value)
        lines.append(f"- {item.key}: {value}")
    return "\n".join(lines)


def _parse_json(text):
    """从模型输出中提取 JSON（容忍 ```json 代码块与前后说明文字）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON：{text[:80]}")
    return json.loads(text[start:end + 1])


def extract_and_save_profile(user_input, assistant_text, store, model):
    """判定本轮是否产生值得长期保存的用户事实，有则写入 store。返回本轮保存的条目列表。

    - 写入格式：store.put(("user_profile",), key, {"value": value})，同 key 覆盖更新
    - 判定/解析失败时静默跳过（记忆写入不能影响主对话）
    """
    try:
        resp = model.invoke([
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=f"用户：{user_input}\n助手：{assistant_text}"),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _parse_json(text)
    except Exception as exc:
        print(f"[长期记忆] 抽取失败，本轮跳过：{exc}")
        return []

    saved = []
    for item in data.get("items", []):
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if not key or not value:
            continue
        store.put(PROFILE_NAMESPACE, key, {"value": value})
        saved.append({"key": key, "value": value})
    if saved:
        print(f"[长期记忆] 已保存：{saved}")
    return saved

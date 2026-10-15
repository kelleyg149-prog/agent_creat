#程序入口：持续对话循环
import time

import openai
from langgraph.types import Command
from langchain_core.messages import AIMessage

from config import config
from graph import graph

#退出指令（大小写不敏感）
EXIT_COMMANDS = {"end", "结束"}

#限流退避参数：429 后最多重试次数与等待秒数（智谱免费档限流宽松，双保险）
RATE_LIMIT_MAX_RETRY = 3
RATE_LIMIT_WAIT_SECONDS = 5


def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_COMMANDS


def get_last_ai_text(result) -> str:
    """提取本轮最后一条有内容的 AI 文本"""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "（模型无文本回复）"


def safe_invoke(payload, config, label="请求"):
    """带异常兜底的 graph.invoke：限流自动退避重试，其余错误友好提示，保证程序不崩溃。

    返回 None 表示本轮请求未完成（调用方决定放弃或重新输入）。
    """
    for attempt in range(RATE_LIMIT_MAX_RETRY + 1):
        try:
            return graph.invoke(payload, config)
        except openai.RateLimitError:
            #限流（429）：等待后重试；超过次数放弃本轮
            if attempt < RATE_LIMIT_MAX_RETRY:
                print(f"提示：{label}过于频繁（限流），{RATE_LIMIT_WAIT_SECONDS} 秒后自动重试（"
                      f"{attempt + 1}/{RATE_LIMIT_MAX_RETRY}）…")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            else:
                print(f"提示：{label}仍被限流，请稍等片刻再继续对话。")
                return None
        except openai.AuthenticationError:
            print("错误：API Key 无效或已过期，请检查 .env 中的 ZAI_API_KEY")
            return None
        except openai.APIConnectionError:
            print("错误：无法连接 API 服务，请检查网络或代理后重试")
            return None
        except openai.BadRequestError as exc:
            print(f"错误：请求格式不正确（{exc}），可尝试重启程序")
            return None
        except Exception as exc:
            print(f"错误：{label}失败（{type(exc).__name__}）：{exc}")
            return None
    return None


def main():
    print("=== 膜数据查询助手 ===")
    print("可直接提问；需要工具时会请求审批（yes/no 或「序号 新编号」）。")
    print("输入 end、END 或 结束 退出。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if is_exit(user_input):
            print("再见！")
            break

        #新的一轮对话：重置重试相关状态（历史消息由 checkpointer 按 thread_id 保留）
        result = safe_invoke({
            "messages": [{"role": "user", "content": user_input}],
            "tool_error": False,
            "retry_count": 0,
            "auto_corrected": False
        }, config)
        if result is None:
            print("（本轮请求未完成，请稍后重新提问）")
            continue

        #处理工具审批：输入保护可能导致 interrupt 多次挂起，循环 resume 直到本轮结束
        exited = False
        while "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            if interrupt_value.get("hint"):
                print("提示：", interrupt_value["hint"])
            print("待审批的工具调用：")
            for i, tc in enumerate(interrupt_value.get("tool_calls", []), 1):
                print(f"  [{i}] {tc['name']}  参数：{tc['args']}")
            answer = input(f"{interrupt_value.get('question', '')}\n> ").strip()
            if is_exit(answer):
                exited = True
                break
            result = safe_invoke(Command(resume=answer), config, label="审批提交")
            if result is None:
                #请求未完成且 interrupt 仍挂起：提示后让用户重新输入，不退出程序
                print("（请求未完成，请重新输入审批指令或稍后再试）")
                continue

        if exited:
            print("再见！")
            break

        print("助手：", get_last_ai_text(result))


if __name__ == "__main__":
    main()

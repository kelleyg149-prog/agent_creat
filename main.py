#程序入口：持续对话循环
from langgraph.types import Command
from langchain_core.messages import AIMessage

from config import config
from graph import graph

#退出指令（大小写不敏感）
EXIT_COMMANDS = {"end", "结束"}


def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_COMMANDS


def get_last_ai_text(result) -> str:
    """提取本轮最后一条有内容的 AI 文本"""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "（模型无文本回复）"


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
        result = graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "tool_error": False,
            "retry_count": 0,
            "auto_corrected": False
        }, config)

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
            result = graph.invoke(Command(resume=answer), config)

        if exited:
            print("再见！")
            break

        print("助手：", get_last_ai_text(result))


if __name__ == "__main__":
    main()

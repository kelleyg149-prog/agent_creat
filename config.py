#配置：API Key、模型实例、运行参数、checkpointer
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from tools import tools

#始终从脚本所在目录加载 .env，避免运行目录不同导致找不到配置
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise RuntimeError("未读取到 MOONSHOT_API_KEY，请检查 .env 文件是否存在且变量名正确")

#初始化月之暗面 Kimi K2.6 模型（Moonshot API 完全兼容 OpenAI 协议）
#base_url：Moonshot 官方接口地址
#extra_body.thinking.type=disabled：关闭思考模式，对齐原 deepseek-chat 的直接问答行为（如需深度推理可移除该参数）
#timeout：单次请求超时（秒），默认值较短，走代理时容易 TLS 握手超时
#max_retries：SDK 层自动重试（指数退避），应对偶发的网络抖动
model = ChatOpenAI(
    model="kimi-k2.6",
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
    timeout=60,
    max_retries=3,
    extra_body={"thinking": {"type": "disabled"}}
)

#绑定工具的模型：在配置层统一创建，供节点工厂函数注入使用
model_with_tool = model.bind_tools(tools)

#失败后最多重试次数（首次调用不计入）
MAX_RETRY = 3

#送给模型的最大消息条数（防止多轮对话历史无限增长撑爆上下文窗口）
MAX_CONTEXT_MESSAGES = 40

#内存 checkpoint，用于跨轮次保留对话历史
checkpointer = InMemorySaver()

#对话线程配置（同一 thread_id 共享历史）
config = {
    "configurable": {
        "thread_id": "1"
    }
}

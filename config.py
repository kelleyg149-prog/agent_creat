#配置：API Key、模型实例、运行参数、checkpointer
import os
import sqlite3
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from tools import tools

#始终从脚本所在目录加载 .env，避免运行目录不同导致找不到配置
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

api_key = os.getenv("ZAI_API_KEY")
if not api_key:
    raise RuntimeError("未读取到 ZAI_API_KEY，请检查 .env 文件是否存在且变量名正确")

#初始化智谱 GLM-5.3 模型（智谱 API 完全兼容 OpenAI 协议）
#base_url：智谱开放平台 OpenAI 兼容接口地址
#extra_body.thinking.type=enabled：GLM-5.3 强制开启思考，不再支持 disabled，否则请求直接报错
#extra_body.reasoning_effort=low：轻量推理档位（GLM-5.3 可选 low/high/max）
#timeout：单次请求超时（秒），默认值较短，走代理时容易 TLS 握手超时
#max_retries：SDK 层自动重试（指数退避），应对偶发的网络抖动
model = ChatOpenAI(
    model="glm-5.3",
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4",
    timeout=60,
    max_retries=3,
    extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
)

#绑定工具的模型：在配置层统一创建，供节点工厂函数注入使用
model_with_tool = model.bind_tools(tools)

#失败后最多重试次数（首次调用不计入）
MAX_RETRY = 3

#送给模型的最大消息条数（防止多轮对话历史无限增长撑爆上下文窗口）
MAX_CONTEXT_MESSAGES = 40

#上下文压缩：超限时保留最近 N 条，更早的消息用 LLM 压缩成摘要（替代硬裁剪，避免丢关键事实）
COMPRESS_KEEP_RECENT = 20

#SQLite checkpointer：对话历史持久化到本地文件（跨进程不丢）
#- 文件放 data/（运行产物，已 gitignore，不入库；含对话内容，删除即清空记忆）
#- check_same_thread=False：LangGraph 内部可能多线程访问 sqlite，Python 默认禁止跨线程
_here_dir = os.path.dirname(os.path.abspath(__file__))
_checkpoint_dir = os.path.join(_here_dir, "data")
os.makedirs(_checkpoint_dir, exist_ok=True)
_checkpoint_conn = sqlite3.connect(
    os.path.join(_checkpoint_dir, "checkpoint.sqlite"),
    check_same_thread=False
)
checkpointer = SqliteSaver(_checkpoint_conn)

#对话线程配置（同一 thread_id 共享历史；换 thread_id = 开新对话）
config = {
    "configurable": {
        "thread_id": "1"
    }
}

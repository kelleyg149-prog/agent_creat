#模块加载
import json
from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()#加载。env文件

api_key = os.getenv("DEEPSEEK_API_KEY")
#执行openAi SDK
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

#membrane_data函数的数据：dict形式
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


def membrane_data(mem_name):
    if mem_name in membrane:
        return membrane[mem_name]
    else:
        return f"未找到膜编号：{mem_name}"


tool = {
    "type": "function",
    "function": {
        "name": "membrane_data",
        "description": "根据膜编号给出相关数据",
        "parameters": {
            "type": "object",
            "properties": {
                "mem_name": {
                    "type": "string",
                    "description": "膜编号，例如 GO-1、GO-2、GO-3"
                }
            },
            "required": ["mem_name"]
        }
    }
}


def calculater(a,b):
    return a+b
calculater_tool={
    "type": "function",
    "function": {
        "name": "calculater",
        "description": "两个数相加",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "第一个数"
                },
                "b": {
                    "type": "number",
                    "description": "第二个数"
                }
            },
            "required": ["a","b"]
        }
    }

}
tools_map = {
    "membrane_data": membrane_data,
    "calculater": calculater
}


class Agent:
    def __init__(self,client,tools,tools_maps,system_prompt,):
        self.client=client
        self.tools=tools
        self.tools_maps=tools_maps
        self.message=[
            {"role":"system",
             "content":system_prompt
            }
        ]

    def agent_run(self,user_input):
        self.message.append(
            {
                "role": "user",
                "content": user_input
            }
        )
        # 是否调用工具and while loop inter
        while True:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.message,
                tools=self.tools
            )

            assistant_message = response.choices[0].message
            self.message.append(assistant_message)
            if not assistant_message.tool_calls:
                return assistant_message.content

            for tool_call in assistant_message.tool_calls:
                arguments = json.loads(
                    tool_call.function.arguments
                )
                name = tool_call.function.name
                function = self.tools_maps.get(name)

                if function is None:
                    result = f"没有找到工具：{name}"
                else:

                    try:
                        result = function(**arguments)

                    except Exception as e:
                        result = f"工具执行失败：{e}"

                self.message.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                })

agent=Agent(
    client,
    [tool,calculater_tool],
    tools_map,
    '你是一名研究材料，尤其是纳滤膜才材料方向的专家助手'
    )
while True:
    user_input=input("你: ")
    if user_input=="exit":
        break

    answer=agent.agent_run(user_input)
    print("AI：",answer)


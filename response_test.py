from dotenv import load_dotenv
from openai import OpenAI
import os

print("1. 程序开始")

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
print("2. API Key是否读取成功：", api_key is not None)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

print("3. Client创建成功")

message = [
    {
        "role": "system",
        "content": "你是一名Python老师。"
    }
]

while True:
    user_input = input("你：")

    if user_input == "exit":
        break

    message.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=message
    )

    answer = response.choices[0].message.content

    print("AI：", answer)

    message.append({
        "role": "assistant",
        "content": answer
    })
print("4. API返回成功")

print(response.choices[0].message.content)
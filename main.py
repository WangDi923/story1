import httpx
from openai import OpenAI

client = OpenAI(
    api_key="sk-957070546df74534b07a2799972a53d0",
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(
        trust_env=False,
        verify=False
    )
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
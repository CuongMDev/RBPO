from openai import OpenAI

client = OpenAI(api_key="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",
                base_url="https://openrouter.ai/api/v1")

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "ping"}],
    max_tokens=10,
)

print(resp)

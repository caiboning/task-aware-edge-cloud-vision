import base64
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set. Add your API key to the .env file.")

if not MODEL:
    raise ValueError("DEEPSEEK_MODEL is not set.")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

image_path = "test.jpg"
query = "What is the main object in this image?"

with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

image_b64 = base64.b64encode(image_bytes).decode("utf-8")
image_data_url = f"data:image/jpeg;base64,{image_b64}"

start_time = time.time()

response = client.chat.completions.create(
    model=MODEL,
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ],
        }
    ],
)

end_time = time.time()

if not response.choices:
    raise RuntimeError("The DeepSeek API returned no completion choices.")

answer = response.choices[0].message.content
if not answer:
    raise RuntimeError(
        "The DeepSeek API returned no text "
        f"(finish_reason={response.choices[0].finish_reason!r})."
    )

print("Query:", query)
print("Answer:", answer)
print("Image bytes:", len(image_bytes))
print("Latency:", end_time - start_time, "seconds")

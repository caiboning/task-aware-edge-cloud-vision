import base64
import io
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from toolbar import config


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
_client = None


def _image_data_url(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = source.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=config.IMAGE_QUALITY
    )
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    return f"data:image/jpeg;base64,{image_b64}"


def ask_deepseek(
    image,
    question,
    client=None,
    return_request_bytes=False
):
    """Send an image and question to the configured DeepSeek cloud model."""
    global _client

    if not DEEPSEEK_MODEL:
        raise ValueError("DEEPSEEK_MODEL is not set.")

    image_data_url = _image_data_url(image)

    if client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Add it to "
                "models/DEEPSEEK/.env or the environment."
            )

        if _client is None:
            _client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        client = _client

    request_data = {
        "model": DEEPSEEK_MODEL,
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a visual question answering assistant. "
                    "Always answer in English, regardless of the language "
                    "used in the question."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ]
    }
    request_bytes = len(
        json.dumps(
            request_data,
            ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")
    )

    start = time.perf_counter()
    response = client.chat.completions.create(**request_data)
    latency = time.perf_counter() - start

    if not response.choices:
        raise RuntimeError("The DeepSeek API returned no completion choices.")

    answer = response.choices[0].message.content
    if not answer:
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise RuntimeError(
                "The DeepSeek response exhausted its "
                f"{DEEPSEEK_MAX_TOKENS}-token output limit before producing "
                "an answer. Increase DEEPSEEK_MAX_TOKENS and retry."
            )

        raise RuntimeError(
            "The DeepSeek API returned no text "
            f"(finish_reason={finish_reason!r})."
        )

    result = (answer.strip(), latency)
    if return_request_bytes:
        return (*result, request_bytes)
    return result

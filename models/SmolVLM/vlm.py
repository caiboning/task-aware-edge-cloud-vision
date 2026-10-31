
import io
import time
from pathlib import Path

from PIL import Image
from ollama import Client

from toolbar import config


# ==============================
# Image Processing
# ==============================

def prepare_image(image):
    """
    Convert image to Ollama-compatible input.

    Accept:
        - str
        - Path
        - PIL.Image.Image
    """

    if isinstance(image, (str, Path)):
        return str(image)

    if isinstance(image, Image.Image):
        buffer = io.BytesIO()

        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=config.IMAGE_QUALITY
        )

        return buffer.getvalue()

    raise TypeError(
        f"Unsupported image type: {type(image)}"
    )


# ==============================
# VLM Inference
# ==============================

def ask_vlm(image, question, client=None):
    """
    Run local VLM inference.

    Returns:
        answer: str
        latency: float (seconds)
    """

    if client is None:
        client = Client(host=config.OLLAMA_HOST)

    image_data = prepare_image(image)

    start = time.perf_counter()

    response = client.chat(
        model=config.VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [image_data]
            }
        ],
        options={
            "temperature": config.VLM_TEMPERATURE,
            "num_predict": config.VLM_NUM_PREDICT,
            "num_ctx": config.VLM_NUM_CTX,
            "repeat_penalty": config.VLM_REPEAT_PENALTY
        }
    )

    latency = time.perf_counter() - start

    answer = response.message.content or ""

    return answer.strip(), latency

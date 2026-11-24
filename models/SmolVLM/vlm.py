import io
import time
from pathlib import Path

from PIL import Image
from ollama import Client

from toolbar import config
from toolbar.profiler import file_size_bytes, payload_size_bytes, save_stage_profile


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

    answer = (response.message.content or "").strip()

    if isinstance(image_data, bytes):
        image_representation = "jpeg_bytes"
        image_bytes = payload_size_bytes(image_data)
    else:
        try:
            image_bytes = file_size_bytes(image_data)
        except (OSError, ValueError):
            image_bytes = None
        image_representation = "image_file" if image_bytes is not None else None

    text_bytes = payload_size_bytes(question)
    save_stage_profile({
        "stage": "smolvlm_inference",
        "model": config.VLM_MODEL,
        "image_input_representation": image_representation,
        "text_input_payload_bytes": text_bytes,
        "image_input_payload_bytes": image_bytes,
        "input_payload_bytes": text_bytes + image_bytes if image_bytes is not None else None,
        "output_payload_bytes": payload_size_bytes(answer),
        "input_tokens": getattr(response, "prompt_eval_count", None),
        "output_tokens": getattr(response, "eval_count", None),
        "latency_ms": round(latency * 1000, 3),
        "local_api_request_bytes": None,
        "local_api_response_bytes": None,
        "peak_ram_bytes": None
    })

    return answer, latency
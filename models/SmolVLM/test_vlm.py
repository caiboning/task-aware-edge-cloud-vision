
import time
from pathlib import Path
from ollama import Client

# ==============================
# Configuration
# ==============================

MODEL = "ahmadwaqar/smolvlm2-256m-video:q8_0"

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "images_test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Find the first supported image
IMAGE_PATH = next(
    (
        file for file in sorted(IMAGE_DIR.iterdir())
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ),
    None
)

# ==============================
# Run VLM
# ==============================

def main():
    if IMAGE_PATH is None:
        print("No supported image found.")
        return

    client = Client(host="http://localhost:11434")

    print("Model:", MODEL)
    print("Image:", IMAGE_PATH)

    # Ask user to enter a question
    question = input("\nQuestion: ").strip()

    if not question:
        print("No question entered.")
        return

    print("\nProcessing...")

    start = time.perf_counter()

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [str(IMAGE_PATH)]
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 80,
            "num_ctx": 2048
        }
    )

    latency = time.perf_counter() - start

    print("\nAnswer:")
    print(response.message.content)

    print(f"\nLatency: {latency:.2f}s")


if __name__ == "__main__":
    main()

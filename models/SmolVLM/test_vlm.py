from pathlib import Path

from toolbar import config
from models.SmolVLM.vlm import ask_vlm


def main():

    image_path = next(
        (
            file for file in sorted(config.IMAGE_DIR.iterdir())
            if file.is_file()
            and file.suffix.lower() in config.IMAGE_EXTENSIONS
        ),
        None
    ) if config.IMAGE_DIR.exists() else None

    if image_path is None:
        print("No supported image found.")
        return

    print("Model:", config.VLM_MODEL)
    print("Image:", image_path)

    question = input("\nQuestion: ").strip()

    if not question:
        print("No question entered.")
        return

    print("\nProcessing...")

    answer, latency = ask_vlm(image_path, question)

    print("\nAnswer:")
    print(answer)

    print(f"\nLatency: {latency:.2f}s")


if __name__ == "__main__":
    main()

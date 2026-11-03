from models.DEEPSEEK.deepseek import ask_deepseek
from toolbar import config


image_paths = sorted(
    path for path in config.IMAGE_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in config.IMAGE_EXTENSIONS
)

if not image_paths:
    raise RuntimeError(f"No test images found in {config.IMAGE_DIR}")

image_path = image_paths[0]
query = "What is the main object in this image?"
answer, latency = ask_deepseek(image_path, query)

print("Image:", image_path)
print("Query:", query)
print("Answer:", answer)
print("Latency:", latency, "seconds")

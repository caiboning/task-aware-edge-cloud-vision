import json
from pathlib import Path

from toolbar import config
from toolbar.logger import get_logger
from models.YOLO.detector import YOLODetector


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "YOLO"

MODELS = {
    "YOLO11n": MODEL_DIR / "yolo11n_ncnn_model",
    "YOLO11s": MODEL_DIR / "yolo11s_ncnn_model",
    "YOLOv8n": MODEL_DIR / "yolov8n_ncnn_model"
}


def main():

    get_logger()

    image_paths = sorted(
        p for p in config.COCO128_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in config.IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    image_path = image_paths[0]

    print(f"Testing image: {image_path.name}")

    for model_name, model_path in MODELS.items():

        print(f"\n========== {model_name} ==========")

        detector = YOLODetector(
            model_path=model_path
        )

        # Warm-up
        detector.detect(image_path)

        # Object Detection
        detections, latency = detector.detect(image_path)

        print(json.dumps(detections, indent=4))
        print(f"Objects: {len(detections)}")
        print(f"Inference Time: {latency * 1000:.2f} ms")


if __name__ == "__main__":
    main()

import json
import time
import cv2
from pathlib import Path
from detector import YOLODetector

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data/coco128/images/train2017"
OUTPUT_DIR = ROOT / "outputs/coco128"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO11n once
detector = YOLODetector(device="cpu", confidence=0.25)

# Find all images
image_paths = sorted([p for p in DATASET_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"Found {len(image_paths)} images.")

if not image_paths:
    raise RuntimeError("No images found in COCO128 dataset.")

# Warm up the model
detector.detect(image_paths[0])

all_results = []
total_inference_time = 0.0
total_image_bytes = 0
total_json_bytes = 0

# Batch evaluation
for index, image_path in enumerate(image_paths, start=1):
    start_time = time.perf_counter()
    results = detector.model.predict(source=str(image_path), conf=detector.confidence, device=detector.device, verbose=False)
    inference_time = time.perf_counter() - start_time

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class": detector.model.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
        })

    payload = {"image": image_path.name, "detections": detections}
    image_bytes = image_path.stat().st_size
    json_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    record = {
        "image": image_path.name,
        "detections": detections,
        "inference_time_ms": round(inference_time * 1000, 2),
        "image_bytes": image_bytes,
        "json_bytes": json_bytes
    }
    all_results.append(record)

    total_inference_time += inference_time
    total_image_bytes += image_bytes
    total_json_bytes += json_bytes

    annotated_image = results[0].plot()
    cv2.imwrite(str(IMAGE_OUTPUT_DIR / image_path.name), annotated_image)

    print(f"[{index}/{len(image_paths)}] {image_path.name} | Objects: {len(detections)} | Time: {inference_time:.3f}s | JSON: {json_bytes} bytes")

# Save detailed results
with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4)

# Save summary
summary = {
    "model": "YOLO11n",
    "device": detector.device,
    "total_images": len(image_paths),
    "total_inference_time_s": round(total_inference_time, 3),
    "average_inference_time_ms": round(total_inference_time * 1000 / len(image_paths), 2),
    "total_image_bytes": total_image_bytes,
    "total_json_bytes": total_json_bytes,
    "data_reduction_percent": round((1 - total_json_bytes / total_image_bytes) * 100, 2)
}

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nCOCO128 Evaluation Completed!")
print(json.dumps(summary, indent=4))

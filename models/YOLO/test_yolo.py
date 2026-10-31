
import json

import cv2

from toolbar import config
from models.YOLO.detector import YOLODetector


# ==============================
# YOLO Batch Test
# ==============================

def main():

    # --------------------------
    # Prepare Directories
    # --------------------------

    config.YOLO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config.YOLO_IMAGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------
    # Find Images
    # --------------------------

    image_paths = sorted(
        p for p in config.COCO128_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in config.IMAGE_EXTENSIONS
    ) if config.COCO128_DIR.exists() else []

    if not image_paths:
        raise RuntimeError(
            f"No images found in {config.COCO128_DIR}"
        )

    print(f"Found {len(image_paths)} images.")

    # --------------------------
    # Initialize YOLO
    # --------------------------

    detector = YOLODetector()

    # --------------------------
    # Warm-up
    # --------------------------

    print("Warming up YOLO...")

    detector.detect(image_paths[0])

    # --------------------------
    # Batch Evaluation
    # --------------------------

    all_results = []

    total_inference_time = 0.0
    total_image_bytes = 0
    total_json_bytes = 0

    for index, image_path in enumerate(
        image_paths, start=1
    ):

        detections, latency, raw_result = (
            detector.detect(
                image_path,
                return_result=True
            )
        )

        payload = {
            "image": image_path.name,
            "detections": detections
        }

        image_bytes = image_path.stat().st_size

        json_bytes = len(
            json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False
            ).encode("utf-8")
        )

        record = {
            "image": image_path.name,
            "detections": detections,
            "inference_time_ms": round(
                latency * 1000, 2
            ),
            "image_bytes": image_bytes,
            "json_bytes": json_bytes
        }

        all_results.append(record)

        total_inference_time += latency
        total_image_bytes += image_bytes
        total_json_bytes += json_bytes

        # Save annotated image
        annotated_image = raw_result.plot()

        cv2.imwrite(
            str(
                config.YOLO_IMAGE_OUTPUT_DIR
                / image_path.name
            ),
            annotated_image
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"Objects: {len(detections)} | "
            f"Time: {latency:.3f}s | "
            f"JSON: {json_bytes} bytes"
        )

    # --------------------------
    # Save Detailed Results
    # --------------------------

    results_path = (
        config.YOLO_OUTPUT_DIR / "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------
    # Summary
    # --------------------------

    total_images = len(image_paths)

    summary = {
        "model": str(config.YOLO_MODEL),
        "device": detector.device,
        "total_images": total_images,
        "total_inference_time_s": round(
            total_inference_time, 3
        ),
        "average_inference_time_ms": round(
            total_inference_time
            * 1000 / total_images, 2
        ),
        "total_image_bytes": total_image_bytes,
        "total_json_bytes": total_json_bytes,
        "data_reduction_percent": round(
            (
                1 - total_json_bytes
                / total_image_bytes
            ) * 100,
            2
        ) if total_image_bytes > 0 else 0
    }

    summary_path = (
        config.YOLO_OUTPUT_DIR / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nCOCO128 Evaluation Completed!")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()

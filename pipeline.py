
import io
import json
import time

from pathlib import Path
from datetime import datetime
from collections import Counter

from PIL import Image

from toolbar import config
from toolbar.logger import get_logger
from toolbar.router import route_task
from toolbar.crop import generate_crops
from toolbar.detection_answer import answer_detection

from models.YOLO.detector import YOLODetector
from models.SmolVLM.vlm import ask_vlm


# ==============================
# Initialization
# ==============================

logger = get_logger()

detector = None


def get_detector():
    """
    Load YOLO once and reuse it.
    """

    global detector

    if detector is None:
        detector = YOLODetector(
            model_path=config.YOLO_MODEL,
            device=config.YOLO_DEVICE,
            confidence=config.YOLO_CONF
        )

    return detector


# ==============================
# Payload Utilities
# ==============================

def calculate_image_bytes(image):

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=config.IMAGE_QUALITY
    )

    return len(buffer.getvalue())


def calculate_json_bytes(data):

    return len(
        json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")
    )


# ==============================
# Save YOLO Visualization
# ==============================

def save_yolo_visualization(
    raw_result,
    relative_path
):

    output_path = (
        config.PIPELINE_YOLO_DIR
        / relative_path.parent
        / (
            f"{relative_path.stem}_"
            f"{relative_path.suffix[1:]}_yolo.jpg"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Ultralytics plot() returns a BGR ndarray
    import cv2

    annotated = raw_result.plot()

    if not cv2.imwrite(
        str(output_path),
        annotated
    ):
        raise OSError(
            f"Failed to save YOLO image: {output_path}"
        )

    return str(output_path)


# ==============================
# Save Crops
# ==============================

def save_crops(crops, relative_path):
    """
    Save every eligible class-specific crop.
    """

    saved_paths = {}

    for class_name, cropped in crops.items():

        safe_class = class_name.replace(
            " ", "_"
        )

        output_path = (
            config.PIPELINE_CROP_DIR
            / relative_path.parent
            / (
                f"{relative_path.stem}_"
                f"{relative_path.suffix[1:]}_"
                f"{safe_class}_crop.jpg"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cropped.save(
            output_path,
            format="JPEG",
            quality=config.IMAGE_QUALITY
        )

        saved_paths[class_name] = str(output_path)

    return saved_paths


# ==============================
# Main Pipeline
# ==============================

def run_pipeline(image_path, question):
    """
    Pipeline:

        1. Load Image
        2. YOLO Detection
        3. Generate Crops
        4. Task Router
        5. Answer
        6. Metrics
    """

    image_path = Path(image_path).resolve()
    total_start = time.perf_counter()

    # --------------------------
    # Step 1: Load Image
    # --------------------------

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    relative_path = image_path.relative_to(
        config.IMAGE_DIR.resolve()
    )

    # --------------------------
    # Step 2: YOLO Detection
    # --------------------------

    yolo_detector = get_detector()

    detections, yolo_latency, raw_result = (
        yolo_detector.detect(
            image,
            return_result=True
        )
    )

    yolo_image_path = save_yolo_visualization(
        raw_result,
        relative_path
    )

    logger.info(
        "YOLO DETECTION | Image: %s | Objects: %d",
        relative_path,
        len(detections)
    )

    # --------------------------
    # Step 3: Generate Crops
    # --------------------------

    crop_start = time.perf_counter()

    crops, crop_info = generate_crops(
        image,
        detections
    )

    crop_paths = save_crops(
        crops,
        relative_path
    )

    crop_latency = (
        time.perf_counter() - crop_start
    )

    for class_name, info in crop_info.items():

        if info["status"] == "SUCCESS":

            logger.info(
                "CROP SUCCESS | Image: %s | "
                "Class: %s | Method: %s",
                relative_path,
                class_name,
                info["method"]
            )

        else:

            logger.warning(
                "CROP FALLBACK | Image: %s | "
                "Class: %s | Reason: %s",
                relative_path,
                class_name,
                info["reason"]
            )

    # --------------------------
    # Step 4: Task Router
    # --------------------------

    router_start = time.perf_counter()

    decision = route_task(
        question,
        detections
    )

    router_latency = (
        time.perf_counter() - router_start
    )

    requested_route = decision["route"]
    task_type = decision["task_type"]
    target_class = decision["target_class"]

    route = requested_route

    # --------------------------
    # Step 5: Answer
    # --------------------------

    answer = None
    vlm_latency = 0.0
    payload_bytes = 0

    used_image = "NONE"
    used_crop_path = None

    if route == "DETECTIONS":

        answer = answer_detection(
            decision,
            detections
        )

        # Simulated YOLO JSON transmission
        payload_bytes = calculate_json_bytes(
            detections
        )

    else:

        target_image = image
        used_image = "FULL_IMAGE"

        if route == "CROP":

            if (
                target_class is not None
                and target_class in crops
            ):

                target_image = crops[target_class]
                used_image = "CROP"

                used_crop_path = crop_paths[
                    target_class
                ]

            else:

                logger.warning(
                    "ROUTE FALLBACK | Image: %s | "
                    "Target: %s | Using FULL_IMAGE",
                    relative_path,
                    target_class
                )

                route = "FULL_IMAGE"

        # Simulated image transmission
        payload_bytes = calculate_image_bytes(
            target_image
        )

        # Local VLM inference
        answer, vlm_latency = ask_vlm(
            target_image,
            question
        )

    # --------------------------
    # Step 6: Validate Answer
    # --------------------------

    if not str(answer).strip():
        raise ValueError(
            "Model returned an empty answer"
        )

    # --------------------------
    # Step 7: Metrics
    # --------------------------

    total_latency = (
        time.perf_counter() - total_start
    )

    metrics = {
        "yolo_latency_ms": round(
            yolo_latency * 1000, 2
        ),
        "crop_latency_ms": round(
            crop_latency * 1000, 2
        ),
        "router_latency_ms": round(
            router_latency * 1000, 2
        ),
        "vlm_latency_ms": round(
            vlm_latency * 1000, 2
        ),
        "total_latency_ms": round(
            total_latency * 1000, 2
        ),
        "payload_bytes": payload_bytes
    }

    # --------------------------
    # Step 8: Output
    # --------------------------

    return {
        "image": str(relative_path),
        "question": question,
        "answer": answer,
        "requested_route": requested_route,
        "route": route,
        "task_type": task_type,
        "target_class": target_class,
        "used_image": used_image,
        "used_crop_path": used_crop_path,
        "detections": detections,
        "yolo_image_path": yolo_image_path,
        "crop_info": crop_info,
        "crop_paths": crop_paths,
        "metrics": metrics,
        "status": "SUCCESS"
    }


# ==============================
# Batch Pipeline Test
# ==============================

def main():

    # --------------------------
    # Step 1: Find Images
    # --------------------------

    image_paths = sorted(
        p for p in config.IMAGE_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ) if config.IMAGE_DIR.exists() else []

    if not image_paths:

        logger.warning(
            "No supported images found in %s",
            config.IMAGE_DIR
        )

        return

    print(f"\nFound {len(image_paths)} images.")
    print("YOLO Model:", config.YOLO_MODEL)
    print("VLM Model:", config.VLM_MODEL)

    # --------------------------
    # Step 2: Question
    # --------------------------

    question = input("\nQuestion: ").strip()

    if not question:

        logger.warning("No question entered")
        return

    # --------------------------
    # Step 3: Output Directories
    # --------------------------

    for directory in [
        config.PIPELINE_OUTPUT_DIR,
        config.PIPELINE_CROP_DIR,
        config.PIPELINE_YOLO_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    all_results = []
    error_results = []

    batch_start = time.perf_counter()

    logger.info("=" * 50)
    logger.info("Pipeline batch started")
    logger.info("Question: %s", question)
    logger.info("Total images: %d", len(image_paths))

    # --------------------------
    # Step 4: Batch Processing
    # --------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image_name = str(
            image_path.relative_to(
                config.IMAGE_DIR
            )
        )

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Processing: {image_name}"
        )

        try:

            result = run_pipeline(
                image_path,
                question
            )

            all_results.append(result)

            logger.info(
                "SUCCESS | Image: %s | "
                "Task: %s | Route: %s | "
                "Latency: %.2f ms",
                image_name,
                result["task_type"],
                result["route"],
                result["metrics"]["total_latency_ms"]
            )

            print("Answer:", result["answer"])
            print("Task Type:", result["task_type"])
            print("Route:", result["route"])
            print("Used Image:", result["used_image"])
            print("Target Class:", result["target_class"])
            print(
                "Generated Crops:",
                len(result["crop_paths"])
            )
            print(
                "Latency:",
                result["metrics"]["total_latency_ms"],
                "ms"
            )
            print(
                "Payload:",
                result["metrics"]["payload_bytes"],
                "bytes"
            )

        except Exception as e:

            error_record = {
                "image": image_name,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

            error_results.append(error_record)

            all_results.append({
                **error_record,
                "status": "FAILED"
            })

            logger.exception(
                "FAILED | Image: %s",
                image_name
            )

            continue

    batch_latency = (
        time.perf_counter() - batch_start
    )

    # --------------------------
    # Step 5: Save Results
    # --------------------------

    with open(
        config.PIPELINE_RESULTS,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    with open(
        config.PIPELINE_ERRORS,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            error_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------
    # Step 6: Summary
    # --------------------------

    successful_results = [
        r for r in all_results
        if r.get("status") == "SUCCESS"
    ]

    total_success = len(successful_results)
    total_failed = len(image_paths) - total_success

    route_counts = dict(
        Counter(
            r["route"]
            for r in successful_results
        )
    )

    task_counts = dict(
        Counter(
            r["task_type"]
            for r in successful_results
        )
    )

    crop_counts = dict(
        Counter(
            info["method"]
            for r in successful_results
            for info in r["crop_info"].values()
            if info["status"] == "SUCCESS"
        )
    )

    crop_fallbacks = sum(
        1
        for r in successful_results
        for info in r["crop_info"].values()
        if info["status"] == "FALLBACK"
    )

    total_generated_crops = sum(
        len(r["crop_paths"])
        for r in successful_results
    )

    total_payload_bytes = sum(
        r["metrics"]["payload_bytes"]
        for r in successful_results
    )

    average_latency = (
        sum(
            r["metrics"]["total_latency_ms"]
            for r in successful_results
        ) / total_success
        if total_success else 0
    )

    average_payload_bytes = (
        total_payload_bytes / total_success
        if total_success else 0
    )

    # --------------------------
    # Step 7: Build Summary
    # --------------------------

    summary = {
        "question": question,
        "total_images": len(image_paths),
        "successful_images": total_success,
        "failed_images": total_failed,
        "success_rate_percent": round(
            total_success / len(image_paths) * 100, 2
        ),
        "route_counts": route_counts,
        "task_counts": task_counts,
        "crop_counts": crop_counts,
        "total_generated_crops": total_generated_crops,
        "crop_fallbacks": crop_fallbacks,
        "average_latency_ms": round(
            average_latency, 2
        ),
        "batch_latency_s": round(
            batch_latency, 3
        ),
        "total_payload_bytes": total_payload_bytes,
        "average_payload_bytes": round(
            average_payload_bytes, 2
        ),
        "yolo_model": str(config.YOLO_MODEL),
        "vlm_model": config.VLM_MODEL,
        "crop_iou_threshold": config.CROP_IOU_THRESHOLD
    }

    # --------------------------
    # Step 8: Save Summary
    # --------------------------

    with open(
        config.PIPELINE_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    logger.info(
        "Batch completed | Success: %d | "
        "Failed: %d | Generated Crops: %d",
        total_success,
        total_failed,
        total_generated_crops
    )

    # --------------------------
    # Step 9: Print Summary
    # --------------------------

    print("\n========== Pipeline Summary ==========")

    print(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nResults saved to:")
    print(config.PIPELINE_OUTPUT_DIR)


if __name__ == "__main__":
    main()

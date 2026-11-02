
import io
import json
import time
import shutil

from pathlib import Path
from datetime import datetime
from collections import Counter

from PIL import Image

from toolbar import config
from toolbar.logger import get_logger
from toolbar.router import route_task
from toolbar.crop import generate_crops
from toolbar.detection_answer import answer_detection

from models.Qwen.llm import LocalLLM
from models.YOLO.detector import YOLODetector
from models.SmolVLM.vlm import ask_vlm
from models.DEEPSEEK.deepseek import DEEPSEEK_MODEL, ask_deepseek


# ==============================
# Initialization
# ==============================

logger = get_logger()

detector = None
llm = None


def get_llm():
    global llm

    if llm is None:
        llm = LocalLLM()

    return llm


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
        2. Qwen Semantic Classification
        3. Task Router
        4. YOLO Detection and Crops when needed
        5. Answer with YOLO, SmolVLM, or DeepSeek
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
    # Step 2: Qwen Classification
    # --------------------------

    llm_start = time.perf_counter()

    task_plan = get_llm().understand_task(
        question
    )

    llm_latency = (
        time.perf_counter() - llm_start
    )

    # --------------------------
    # Step 3: Task Router
    # --------------------------

    router_start = time.perf_counter()

    decisions = [
        route_task(task)
        for task in task_plan["tasks"]
    ]

    router_latency = (
        time.perf_counter() - router_start
    )

    detections = []
    yolo_latency = 0.0
    yolo_image_path = None
    crops = {}
    crop_info = {}
    crop_paths = {}
    crop_latency = 0.0

    needs_yolo = any(
        decision["route"] in ("DETECTIONS", "CROP")
        for decision in decisions
    )

    if needs_yolo:
        # --------------------------
        # Step 4: YOLO Detection
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
        # Step 5: Generate Crops
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
    # Step 6: Answer
    # --------------------------

    task_results = []

    vlm_latency = 0.0
    cloud_latency = 0.0
    payload_bytes = 0

    used_image = "NONE"
    used_crop_path = None

    for decision in decisions:

        requested = decision["route"]
        route = requested

        target_class = decision["target_class"]
        task_type = decision["task_type"]

        task_status = "SUCCESS"
        current_crop_path = None
        current_used_image = "NONE"
        current_payload = 0

        if route == "DETECTIONS":

            answer_part = answer_detection(
                decision,
                detections
            )

            current_payload = calculate_json_bytes(
                detections
            )

        elif route == "CLOUD_ANSWER":

            current_used_image = "FULL_IMAGE"
            current_payload = calculate_image_bytes(image)

            if len(decisions) == 1:
                cloud_question = question
            else:
                cloud_question = (
                    "Answer only this part of the "
                    "user's question: "
                    f"task={decision['semantic_type']}, "
                    f"target={decision['target']}, "
                    f"attribute={decision['attribute']}. "
                    f"Original question: {question}"
                )

            answer_part, latency = ask_deepseek(
                image,
                cloud_question
            )

            cloud_latency += latency

        else:

            target_image = image
            current_used_image = "FULL_IMAGE"

            if route == "CROP":

                if (
                    target_class is not None
                    and target_class in crops
                ):

                    target_image = crops[target_class]
                    current_used_image = "CROP"

                    current_crop_path = crop_paths[
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

            current_payload = calculate_image_bytes(
                target_image
            )

            # Build VLM question.
            if len(decisions) == 1:

                vlm_question = question

            else:

                vlm_question = (
                    "Answer only this part of the "
                    "user's question: "
                    f"task={decision['semantic_type']}, "
                    f"target={decision['target']}, "
                    f"attribute={decision['attribute']}. "
                    f"Original question: {question}"
                )

            if decision["semantic_type"] == "ocr":
                vlm_question = (
                    "Transcribe only text that is clearly readable "
                    "in the image. Do not describe the image or infer "
                    "text that cannot be read. If there is no readable "
                    "text, reply exactly: "
                    "\"No readable text is visible in the image.\""
                )

            answer_part, latency = ask_vlm(
                target_image,
                vlm_question
            )

            vlm_latency += latency

        # --------------------------
        # Validate Task Answer
        # --------------------------

        if not str(answer_part).strip():
            raise ValueError(
                "Model returned an empty answer"
            )

        payload_bytes += current_payload

        if current_used_image != "NONE":
            used_image = current_used_image

        if current_crop_path is not None:
            used_crop_path = current_crop_path

        task_results.append({
            "task_type": task_type,
            "semantic_type": decision["semantic_type"],
            "route": route,
            "requested_route": requested,
            "target_class": target_class,
            "answer": answer_part,
            "status": task_status
        })

    # --------------------------
    # Combine Task Answers
    # --------------------------

    if len(task_results) == 1:

        answer = task_results[0]["answer"]

    else:

        answer = "\n".join(
            f"{i}. {item['answer']}"
            for i, item in enumerate(
                task_results,
                start=1
            )
        )

    if len(task_results) == 1:

        requested_route = (
            task_results[0]["requested_route"]
        )

        route = task_results[0]["route"]
        task_type = task_results[0]["task_type"]

        target_class = (
            task_results[0]["target_class"]
        )

    else:

        requested_route = "MULTI_TASK"
        route = "MULTI_TASK"
        task_type = "MULTI_TASK"
        target_class = None

    # --------------------------
    # Step 7: Validate Answer
    # --------------------------

    if not str(answer).strip():
        raise ValueError(
            "Model returned an empty answer"
        )

    # --------------------------
    # Step 8: Metrics
    # --------------------------

    total_latency = (
        time.perf_counter() - total_start
    )

    metrics = {
        "llm_latency_ms": round(
            llm_latency * 1000, 2
        ),
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
        "cloud_latency_ms": round(
            cloud_latency * 1000, 2
        ),
        "total_latency_ms": round(
            total_latency * 1000, 2
        ),
        "payload_bytes": payload_bytes
    }

    # --------------------------
    # Step 9: Output
    # --------------------------

    statuses = [
        task["status"]
        for task in task_results
    ]

    if all(s == "SUCCESS" for s in statuses):
        status = "SUCCESS"

    elif any(s == "SUCCESS" for s in statuses):
        status = "PARTIAL_SUCCESS"

    else:
        status = "FAILED"

    return {
        "image": str(relative_path),
        "question": question,
        "task_plan": task_plan,
        "decisions": decisions,
        "task_results": task_results,
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
        "status": status
    }

# ==============================
# Clear Previous Outputs
# ==============================

def clear_previous_outputs():

    output_dir = config.PIPELINE_OUTPUT_DIR.resolve()

    project_root = Path(__file__).resolve().parent

    allowed_root = (
        project_root / "outputs"
    ).resolve()

    # Safety check
    if (
        output_dir == allowed_root
        or allowed_root not in output_dir.parents
    ):
        raise ValueError(
            f"Unsafe output directory: {output_dir}"
        )

    if not output_dir.exists():
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        return

    # Clear files and subdirectories,
    # but keep active log files.
    for item in output_dir.iterdir():

        if item.is_file():

            if item.suffix.lower() == ".log":
                continue

            item.unlink()

        elif item.is_dir():

            shutil.rmtree(item)

    print(
        f"Previous outputs cleared: {output_dir}"
    )

# ==============================
# Batch Pipeline Test
# ==============================

def main():

    # --------------------------
    # Clear Previous Outputs
    # --------------------------

    clear_previous_outputs()

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
    print("DeepSeek Model:", DEEPSEEK_MODEL)

    # --------------------------
    # Step 2: Question
    # --------------------------

    try:
        question = input("\nQuestion: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nExiting pipeline.")
        return

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
        "deepseek_model": DEEPSEEK_MODEL,
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

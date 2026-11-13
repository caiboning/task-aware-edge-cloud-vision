
import argparse
import json
import re
import time

from pathlib import Path
from datetime import datetime
from collections import Counter

from PIL import Image

from toolbar import config
from toolbar.logger import get_logger
from toolbar.router import route_task
from toolbar.crop import generate_crops
from toolbar.detection_answer import (
    answer_detection,
    answer_missing_target
)

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


def get_report_path(report_name):
    """Create a unique report path for this pipeline run."""

    report_name = Path(report_name)
    if (
        report_name.name != str(report_name)
        or report_name.stem in {"", ".", ".."}
    ):
        raise ValueError(
            "Report name must be a filename or filename stem, "
            "not a path."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = report_name.stem
    report_path = (
        config.PIPELINE_TEXT.parent
        / f"{report_stem}_{timestamp}.txt"
    )

    suffix = 1
    while report_path.exists():
        report_path = (
            config.PIPELINE_TEXT.parent
            / f"{report_stem}_{timestamp}_{suffix}.txt"
        )
        suffix += 1

    return report_path


def route_models(requested_route, route):
    """Return the answering/inference models used for a task route."""

    if requested_route == "CROP" and route == "CROP":
        return ["YOLO", "Cloud (DeepSeek)"]
    if requested_route == "CROP" and route == "CLOUD_ANSWER":
        return ["YOLO", "Cloud (DeepSeek)"]
    if route == "DETECTIONS":
        return ["YOLO"]
    if route == "CLOUD_ANSWER":
        return ["Cloud (DeepSeek)"]
    if route == "FULL_IMAGE":
        return ["SmolVLM"]
    return []


def format_route(route, models, requested_route=None):
    """Format a route with its model names and any fallback information."""

    formatted = route
    if models:
        formatted += f" -> {' + '.join(models)}"
    if requested_route and requested_route != route:
        formatted += f" [fallback from {requested_route}]"
    return formatted


def write_text_report(results, summary, report_path):
    """Write a readable batch report alongside the JSON outputs."""

    lines = [
        "Task-Aware Vision Pipeline Report",
        "=" * 34,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Question: {summary['question']}",
        "",
        "Batch Summary",
        "-" * 13,
        f"Images: {summary['total_images']}",
        f"Successful: {summary['successful_images']}",
        f"Failed: {summary['failed_images']}",
        f"Success rate: {summary['success_rate_percent']}%",
        f"Routes: {json.dumps(summary['route_counts'], ensure_ascii=False)}",
        f"Task types: {json.dumps(summary['task_counts'], ensure_ascii=False)}",
        f"Average latency: {summary['average_latency_ms']} ms",
        "Edge-to-cloud payload (estimated request body)",
        "-" * 48,
        f"Total: {summary['total_edge_to_cloud_bytes']} bytes",
        f"Average per image: {summary['average_edge_to_cloud_bytes']} bytes",
        "Local Qwen/YOLO/SmolVLM processing contributes 0 bytes. "
        "Crop routes upload the detected crop to DeepSeek; the estimate "
        "excludes HTTP headers, TLS overhead, and response traffic.",
        f"Generated crops: {summary['total_generated_crops']}",
        f"Crop fallbacks: {summary['crop_fallbacks']}",
        "",
        "Models",
        "-" * 6,
        f"Qwen: {summary['qwen_model']}",
        f"YOLO: {Path(summary['yolo_model']).name}",
        f"SmolVLM: {summary['vlm_model']}",
        f"DeepSeek: {summary['deepseek_model']}",
        "",
        "Per-Image Results",
        "-" * 17,
    ]

    for index, result in enumerate(results, start=1):
        lines.extend([
            "",
            f"[{index}/{len(results)}] {result['image']}",
            f"Status: {result.get('status', 'UNKNOWN')}",
        ])

        if result.get("status") == "FAILED":
            lines.extend([
                f"Error: {result.get('error_type', 'Error')}",
                f"Details: {result.get('error_message', 'No details')}",
            ])
            continue

        metrics = result["metrics"]
        lines.extend([
            f"Task: {result['task_type']}",
            f"Route: {format_route(result['route'], result.get('models', []), result.get('requested_route'))}",
            f"Target: {result.get('target_class') or 'N/A'}",
            f"Answer: {result['answer']}",
        ])
        if result.get("used_image") != "NONE":
            lines.append(f"Image used: {result['used_image']}")
        if result.get("used_crop_path"):
            lines.append(f"Crop output: {result['used_crop_path']}")

        task_results = result.get("task_results", [])
        if len(task_results) > 1:
            lines.append("Task answers:")
            for task_index, task in enumerate(task_results, start=1):
                lines.append(
                    f"  {task_index}. {task['task_type']} "
                    f"({format_route(task['route'], task.get('models', []), task.get('requested_route'))}): "
                    f"{task['answer']}"
                )
                if task.get("used_image") != "NONE":
                    lines.append(
                        f"     Image used: {task['used_image']}"
                    )
                if task.get("used_crop_path"):
                    lines.append(
                        f"     Crop output: {task['used_crop_path']}"
                    )

        lines.extend([
            f"Latency: {metrics['total_latency_ms']} ms "
            f"(Qwen {metrics.get('llm_latency_ms', 0)} ms, "
            f"YOLO {metrics.get('yolo_latency_ms', 0)} ms, "
            f"SmolVLM {metrics.get('vlm_latency_ms', 0)} ms, "
            f"Cloud {metrics.get('cloud_latency_ms', 0)} ms)",
            "Edge-to-cloud payload (estimated request body): "
            f"{metrics['edge_to_cloud_bytes']} bytes",
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
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
        route_task(task, question=question)
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
    edge_to_cloud_bytes = 0

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

        if route == "DETECTIONS":

            answer_part = answer_detection(
                decision,
                detections
            )

        elif route == "CROP" and target_class is not None and not any(
            detection["class"].lower() == target_class
            for detection in detections
        ):

            answer_part = answer_missing_target(
                decision,
                detections
            )
            route = "DETECTIONS"

        elif route == "CLOUD_ANSWER":

            current_used_image = "FULL_IMAGE"

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

            answer_part, latency, request_body_bytes = ask_deepseek(
                image,
                cloud_question,
                return_request_bytes=True
            )

            cloud_latency += latency
            edge_to_cloud_bytes += request_body_bytes

        else:

            target_image = image
            current_used_image = "FULL_IMAGE"
            crop_available = False

            if route == "CROP":

                if (
                    target_class is not None
                    and target_class in crops
                ):

                    target_image = crops[target_class]
                    current_used_image = "CROP"
                    crop_available = True

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

                    route = "CLOUD_ANSWER"

            if requested == "CROP":
                if decision["semantic_type"] == "ocr":
                    cloud_question = (
                        f"Read only clearly visible text on the {target_class} "
                        "in the provided image. Ignore text outside the target. "
                        "Do not describe the image or infer unreadable text. "
                        "If no text on the target is readable, say so."
                    )
                elif (
                    decision["semantic_type"] == "attribute"
                    and (
                        str(decision.get("attribute") or "").lower()
                        in {"color", "colour"}
                        or re.search(
                            r"\bcolou?rs?\b",
                            question,
                            re.IGNORECASE
                        )
                    )
                ):
                    cloud_question = (
                        f"Look at the {target_class} in the provided image and "
                        "report the main color of its own material or surface. "
                        "Ignore text, route numbers, signs, logos, labels, "
                        "windows, lights, reflections, and the background. "
                        "Do not read or repeat text. Give only a short color "
                        "description; if uncertain, answer 'unclear'."
                    )
                else:
                    cloud_question = (
                        "Answer the user's question using only the provided "
                        f"image of the {target_class}. Original question: "
                        f"{question}"
                    )
                if not crop_available:
                    cloud_question += (
                        " The requested target crop was unavailable, so this is "
                        "the full image; focus only on the requested target."
                    )
                answer_part, latency, request_body_bytes = ask_deepseek(
                    target_image,
                    cloud_question,
                    return_request_bytes=True
                )
                cloud_latency += latency
                edge_to_cloud_bytes += request_body_bytes
            else:
                if len(decisions) == 1:
                    vlm_question = question
                else:
                    vlm_question = (
                        "Answer only this part of the user's question: "
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
                elif (
                    decision["semantic_type"] == "attribute"
                    and (
                        str(decision.get("attribute") or "").lower()
                        in {"color", "colour"}
                        or re.search(
                            r"\bcolou?rs?\b",
                            question,
                            re.IGNORECASE
                        )
                    )
                ):
                    target_name = decision.get("target") or "object"
                    vlm_question = (
                        f"Look at the {target_name} in the image and answer "
                        "the color question. Report the main color of the "
                        "object's own material or surface. Ignore text, signs, "
                        "labels, lights, reflections, and background. Reply "
                        "with a short color description or 'unclear'."
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

        used_models = route_models(requested, route)

        if current_used_image != "NONE":
            used_image = current_used_image

        if current_crop_path is not None:
            used_crop_path = current_crop_path

        task_results.append({
            "task_type": task_type,
            "semantic_type": decision["semantic_type"],
            "route": route,
            "requested_route": requested,
            "models": used_models,
            "target_class": target_class,
            "answer": answer_part,
            "status": task_status,
            "used_image": current_used_image,
            "used_crop_path": current_crop_path
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

    models = list(dict.fromkeys(
        model
        for task in task_results
        for model in task["models"]
    ))

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
        "edge_to_cloud_bytes": edge_to_cloud_bytes
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
        "models": models,
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
# Batch Pipeline Test
# ==============================

def main(argv=None):

    # --------------------------
    # Command-Line Options
    # --------------------------

    parser = argparse.ArgumentParser(
        description="Run the task-aware vision pipeline."
    )
    parser.add_argument(
        "--report-name",
        default="results",
        help=(
            "Custom report filename or stem. A timestamp is added "
            "to preserve reports from previous runs."
        )
    )
    args = parser.parse_args(argv)
    report_path = get_report_path(args.report_name)

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
                format_route(
                    result["route"],
                    result["models"],
                    result["requested_route"]
                ),
                result["metrics"]["total_latency_ms"]
            )

            print("Answer:", result["answer"])
            print("Task Type:", result["task_type"])
            print(
                "Route:",
                format_route(
                    result["route"],
                    result["models"],
                    result["requested_route"]
                )
            )
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
                "Edge-to-cloud payload (estimated request body):",
                result["metrics"]["edge_to_cloud_bytes"],
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

    average_latency = (
        sum(
            r["metrics"]["total_latency_ms"]
            for r in successful_results
        ) / total_success
        if total_success else 0
    )

    total_edge_to_cloud_bytes = sum(
        result["metrics"]["edge_to_cloud_bytes"]
        for result in successful_results
    )
    average_edge_to_cloud_bytes = (
        total_edge_to_cloud_bytes / total_success
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
        "total_edge_to_cloud_bytes": total_edge_to_cloud_bytes,
        "average_edge_to_cloud_bytes": round(
            average_edge_to_cloud_bytes, 2
        ),
        "qwen_model": "Qwen3-0.6B",
        "yolo_model": str(config.YOLO_MODEL),
        "vlm_model": config.VLM_MODEL,
        "deepseek_model": DEEPSEEK_MODEL,
        "crop_iou_threshold": config.CROP_IOU_THRESHOLD
    }

    write_text_report(
        all_results,
        summary,
        report_path
    )

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
    print("Readable report:", report_path)


if __name__ == "__main__":
    main()

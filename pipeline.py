
import json
import re
import time

from pathlib import Path
from datetime import datetime
from collections import Counter

from PIL import Image

from toolbar import config
from toolbar.logger import get_logger
from toolbar.profiler import profile_pipeline
from toolbar.model_profiles import ncnn_profile
from toolbar.payload_totals import model_key, yolo_run_groups
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

detectors = {}
llm = None


def get_llm():
    global llm

    if llm is None:
        llm = LocalLLM()

    return llm


def get_detector(model_path):
    if model_path not in detectors:
        detectors[model_path] = YOLODetector(
            model_path=model_path,
            device=config.YOLO_DEVICE,
            confidence=config.YOLO_CONF
        )

    return detectors[model_path]


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


def format_total_payload(metrics):
    total_bytes = metrics.get("edge_cloud_total_bytes")
    size = f"{total_bytes} bytes" if total_bytes is not None else "N/A"
    return f"Edge-to-cloud payload (actual request + response bodies): {size}"


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
        f"Average end-to-end latency: {summary['average_latency_ms']} ms",
        f"Average estimated Edge-to-Cloud request bytes: "
        f"{summary['average_edge_to_cloud_bytes']} bytes",
        "Local Qwen/YOLO/SmolVLM processing contributes 0 bytes. "
        "Crop routes upload the detected crop to DeepSeek; payload is an "
        "estimated serialized request body, excluding HTTP headers, TLS "
        "overhead, and response traffic.",
        "",
        "Models",
        "-" * 6,
        f"Qwen: {summary['qwen_model']}",
        f"YOLO: {Path(summary['yolo_model']).name}",
        f"SmolVLM: {summary['vlm_model']}",
        f"DeepSeek: {summary['deepseek_model']}",
    ]
    lines.extend(
        f"  {stage}: {latency} ms"
        for stage, latency in summary[
            "average_component_latency_ms"
        ].items()
    )
    groups = yolo_run_groups(results)
    if groups:
        lines.extend(["", "YOLO Model Parameters and Batch Latency"])
        for model_path in (config.YOLO_MODEL, *config.YOLO_COMPARISON_MODELS):
            model = model_path.name
            if model not in groups:
                continue
            key = model_key({"stage": "yolo_detection", "model": model})
            profile = ncnn_profile(key, model_path)
            runs = groups[model]
            lines.extend([
                f"{model}",
                f"  Parameters: {profile.get('parameter_count') or 'N/A'}",
                f"  Parameter basis: {profile.get('parameter_count_basis', 'N/A')}",
                f"  Format: {profile.get('model_format', 'N/A')}",
                f"  Input resolution: {profile.get('input_resolution') or 'N/A'}",
                f"  Weights: {profile.get('weight_size_bytes') or 'N/A'} bytes",
                f"  Deployment size: {profile.get('total_model_size_bytes') or 'N/A'} bytes",
                f"  Device: {config.YOLO_DEVICE}, confidence threshold: {config.YOLO_CONF}",
                f"  {yolo_latency_summary(runs)}",
            ])

    lines.extend([
        "",
        "Per-Image Results",
        "-" * 17,
    ])

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
            f"(image load {metrics['image_load_latency_ms']} ms, "
            f"Qwen load {metrics['llm_model_load_latency_ms']} ms, "
            f"Qwen inference {metrics['llm_latency_ms']} ms, "
            f"router {metrics['router_latency_ms']} ms, "
            f"YOLO load {metrics['yolo_model_load_latency_ms']} ms, "
            f"YOLO inference {metrics['yolo_latency_ms']} ms, "
            f"crop {metrics['crop_latency_ms']} ms, "
            f"SmolVLM setup {metrics['vlm_preparation_latency_ms']} ms, "
            f"SmolVLM call {metrics['vlm_latency_ms']} ms, "
            f"DeepSeek setup {metrics['cloud_preparation_latency_ms']} ms, "
            f"DeepSeek API {metrics['cloud_latency_ms']} ms, "
            f"other {metrics['other_latency_ms']} ms)",
            "Edge-to-cloud payload (estimated request body): "
            f"{metrics['edge_to_cloud_bytes']} bytes",
            format_total_payload(metrics),
        ])
        if result.get("yolo_model"):
            for model_result in (result, *result.get("yolo_comparisons", [])):
                lines.extend(yolo_result_lines(model_result))
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
@profile_pipeline
def run_pipeline(image_path, question, *, yolo_model=None, task_plan=None):
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

    image_load_start = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_load_latency = time.perf_counter() - image_load_start

    relative_path = image_path.relative_to(
        config.IMAGE_DIR.resolve()
    )

    # --------------------------
    # Step 2: Qwen Classification
    # --------------------------

    llm_model_load_latency = 0.0
    llm_latency = 0.0
    if task_plan is None:
        if llm is None:
            llm_model_load_start = time.perf_counter()
            llm_model = get_llm()
            llm_model_load_latency = time.perf_counter() - llm_model_load_start
        else:
            llm_model = llm

        llm_start = time.perf_counter()
        task_plan = llm_model.understand_task(question)
        llm_latency = time.perf_counter() - llm_start

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
    yolo_model_load_latency = 0.0
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
        yolo_model = yolo_model or config.YOLO_MODEL
        artifact_path = relative_path
        if yolo_model != config.YOLO_MODEL:
            artifact_path = Path(yolo_model.name) / relative_path

        if yolo_model not in detectors:
            yolo_model_load_start = time.perf_counter()
            yolo_detector = get_detector(yolo_model)
            yolo_model_load_latency = time.perf_counter() - yolo_model_load_start
        else:
            yolo_detector = detectors[yolo_model]

        detections, yolo_latency, raw_result = (
            yolo_detector.detect(
                image,
                return_result=True
            )
        )

        yolo_image_path = save_yolo_visualization(
            raw_result,
            artifact_path
        )

        logger.info(
            "YOLO DETECTION | Image: %s | Objects: %d",
            relative_path,
            len(detections)
        )

        if any(decision["route"] == "CROP" for decision in decisions):
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
                artifact_path
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
    vlm_preparation_latency = 0.0
    cloud_latency = 0.0
    cloud_preparation_latency = 0.0
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

            cloud_call_start = time.perf_counter()
            answer_part, latency, request_body_bytes = ask_deepseek(
                image,
                cloud_question,
                return_request_bytes=True
            )
            cloud_preparation_latency += max(
                0.0,
                time.perf_counter() - cloud_call_start - latency
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
                cloud_call_start = time.perf_counter()
                answer_part, latency, request_body_bytes = ask_deepseek(
                    target_image,
                    cloud_question,
                    return_request_bytes=True
                )
                cloud_preparation_latency += max(
                    0.0,
                    time.perf_counter() - cloud_call_start - latency
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
                        "unreadable text. If there is no readable text, reply exactly: "
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

                vlm_call_start = time.perf_counter()
                answer_part, latency = ask_vlm(
                    target_image,
                    vlm_question
                )
                vlm_preparation_latency += max(
                    0.0,
                    time.perf_counter() - vlm_call_start - latency
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

    measured_stage_latency = sum((
        image_load_latency,
        llm_model_load_latency,
        llm_latency,
        router_latency,
        yolo_model_load_latency,
        yolo_latency,
        crop_latency,
        vlm_preparation_latency,
        vlm_latency,
        cloud_preparation_latency,
        cloud_latency
    ))

    metrics = {
        "image_load_latency_ms": round(
            image_load_latency * 1000, 2
        ),
        "llm_model_load_latency_ms": round(
            llm_model_load_latency * 1000, 2
        ),
        "llm_latency_ms": round(
            llm_latency * 1000, 2
        ),
        "yolo_model_load_latency_ms": round(
            yolo_model_load_latency * 1000, 2
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
        "vlm_preparation_latency_ms": round(
            vlm_preparation_latency * 1000, 2
        ),
        "cloud_latency_ms": round(
            cloud_latency * 1000, 2
        ),
        "cloud_preparation_latency_ms": round(
            cloud_preparation_latency * 1000, 2
        ),
        "total_latency_ms": round(
            total_latency * 1000, 2
        ),
        "other_latency_ms": round(
            max(0.0, total_latency - measured_stage_latency) * 1000,
            2
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
        "yolo_model": yolo_model.name if needs_yolo else None,
        "yolo_image_path": yolo_image_path,
        "crop_info": crop_info,
        "crop_paths": crop_paths,
        "metrics": metrics,
        "status": status
    }

# ==============================
# Batch Pipeline Runner
# ==============================

def yolo_latency_summary(runs):
    successful = [run for run in runs if run.get("status") == "SUCCESS"]
    if not successful:
        return f"0/{len(runs)} successful; latency N/A"
    yolo_ms = sum(run["metrics"]["yolo_latency_ms"] for run in successful)
    end_to_end_ms = sum(run["metrics"]["total_latency_ms"] for run in successful)
    count = len(successful)
    return (
        f"{count}/{len(runs)} successful; YOLO total {yolo_ms:.2f} ms, "
        f"average {yolo_ms / count:.2f} ms; end-to-end total {end_to_end_ms:.2f} ms, "
        f"average {end_to_end_ms / count:.2f} ms"
    )


def yolo_result_lines(result):
    yield f"YOLO model: {result['yolo_model']}"
    if "error_message" in result:
        yield f"  Failed: {result['error_message']}"
        return

    yield f"  Route: {result['route']}"
    yield f"  Answer: {result['answer']}"
    yield f"  Detected objects: {len(result['detections'])}"
    yield f"  YOLO model load: {result['metrics']['yolo_model_load_latency_ms']} ms"
    yield f"  YOLO inference: {result['metrics']['yolo_latency_ms']} ms"
    yield f"  End-to-end latency: {result['metrics']['total_latency_ms']} ms"
    yield f"  Detection image: {result['yolo_image_path']}"
    if result.get("used_crop_path"):
        yield f"  Crop output: {result['used_crop_path']}"


def compare_yolo_models(image_path, question, result):
    comparisons = []
    for model_path in config.YOLO_COMPARISON_MODELS:
        if model_path == config.YOLO_MODEL:
            continue
        try:
            comparison = run_pipeline(
                image_path, question, yolo_model=model_path,
                task_plan=result["task_plan"],
            )
        except Exception as error:
            logger.exception("YOLO comparison failed | Model: %s", model_path.name)
            comparison = {
                "yolo_model": model_path.name,
                "status": "FAILED",
                "error_message": str(error),
            }
        if "runtime_metrics" in comparison:
            # Attribute the shared classification to each logical model variant.
            shared = [
                stage for stage in result["runtime_metrics"]
                if stage["stage"] == "qwen_classification"
            ]
            comparison["runtime_metrics"] = shared + comparison["runtime_metrics"]
        comparisons.append(comparison)
    return comparisons


def run_batch(question, image_paths, report_name="results"):
    image_paths = list(image_paths)
    report_path = get_report_path(report_name)
    question = question.strip() if question else ""

    if not image_paths:
        logger.warning(
            "No supported images found in %s",
            config.IMAGE_DIR
        )
        return

    print(f"\nFound {len(image_paths)} images.")
    print("YOLO models:", ", ".join(
        path.name for path in (config.YOLO_MODEL, *config.YOLO_COMPARISON_MODELS)
    ))
    print("VLM Model:", config.VLM_MODEL)
    print("DeepSeek Model:", DEEPSEEK_MODEL)

    if not question:
        logger.warning("No question entered")
        return

    # --------------------------
    # Step 1: Output Directories
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

    logger.info("=" * 50)
    logger.info("Pipeline batch started")
    logger.info("Question: %s", question)
    logger.info("Total images: %d", len(image_paths))

    # --------------------------
    # Step 2: Batch Processing
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

            if result["yolo_model"]:
                result["yolo_comparisons"] = compare_yolo_models(
                    image_path, question, result
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
            print(format_total_payload(result["metrics"]))
            if result["yolo_model"]:
                for model_result in (result, *result["yolo_comparisons"]):
                    print("\n".join(yolo_result_lines(model_result)))

        except Exception as e:

            error_record = {
                "image": image_name,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

            all_results.append({
                **error_record,
                "status": "FAILED"
            })

            logger.exception(
                "FAILED | Image: %s",
                image_name
            )

            continue

    # --------------------------
    # Step 3: Save Results
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

    # --------------------------
    # Step 4: Summary
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
    latency_components = {
        "image_load": "image_load_latency_ms",
        "qwen_model_load": "llm_model_load_latency_ms",
        "qwen_inference": "llm_latency_ms",
        "router": "router_latency_ms",
        "yolo_model_load": "yolo_model_load_latency_ms",
        "yolo_inference": "yolo_latency_ms",
        "crop_generation": "crop_latency_ms",
        "smolvlm_preparation": "vlm_preparation_latency_ms",
        "smolvlm_call": "vlm_latency_ms",
        "deepseek_preparation": "cloud_preparation_latency_ms",
        "deepseek_api": "cloud_latency_ms",
        "other_pipeline": "other_latency_ms",
        "total": "total_latency_ms"
    }
    average_component_latency = {
        stage: round(
            sum(
                result["metrics"][metric]
                for result in successful_results
            ) / total_success,
            2
        ) if total_success else 0
        for stage, metric in latency_components.items()
    }

    total_edge_to_cloud_bytes = sum(
        result["metrics"]["edge_to_cloud_bytes"]
        for result in successful_results
    )
    average_edge_to_cloud_bytes = (
        total_edge_to_cloud_bytes / total_success
        if total_success else 0
    )
    # --------------------------
    # Step 5: Build Summary
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
        "average_component_latency_ms": average_component_latency,
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

    logger.info(
        "Batch completed | Success: %d | "
        "Failed: %d | Generated Crops: %d",
        total_success,
        total_failed,
        total_generated_crops
    )

    # --------------------------
    # Step 7: Print Summary
    # --------------------------

    print("\n========== Pipeline Summary ==========")

    display_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"total_generated_crops", "crop_fallbacks"}
    }
    print(
        json.dumps(
            display_summary,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nResults saved to:")
    print(config.PIPELINE_OUTPUT_DIR)
    print("Readable report:", report_path)
    return summary

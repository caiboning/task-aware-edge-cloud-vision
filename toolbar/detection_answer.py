import time
from collections import Counter
from functools import wraps

from toolbar.profiler import json_size_bytes, payload_size_bytes, save_stage_profile


def _profile_answer(function):
    @wraps(function)
    def wrapped(decision, detections):
        started = time.perf_counter()
        answer = function(decision, detections)
        latency_ms = (time.perf_counter() - started) * 1000

        save_stage_profile({
            "stage": "local_detection_answer",
            "operation": function.__name__,
            "input_payload_bytes": json_size_bytes({
                "decision": decision,
                "detections": detections
            }),
            "output_payload_bytes": payload_size_bytes(answer),
            "latency_ms": round(latency_ms, 3)
        })
        return answer

    return wrapped


# ==============================
# Detection Answer
# ==============================

@_profile_answer
def answer_detection(decision, detections):
    """
    Generate a natural-language answer
    from YOLO detection results.

    Supported task types:
        EXISTENCE
        COUNT
        LIST

    Args:
        decision: Router output dictionary
        detections: YOLO detection list

    Returns:
        str: Natural-language answer
    """

    task_type = decision.get("task_type")
    target_class = decision.get("target_class")

    # Count detected objects by class
    counts = Counter(
        d["class"].lower()
        for d in detections
    )

    # --------------------------
    # 1. Object Existence
    # --------------------------

    if task_type == "EXISTENCE":

        if target_class is None:
            raise ValueError(
                "EXISTENCE requires target_class"
            )

        count = counts.get(target_class, 0)

        if count > 0:
            return (
                f"Yes, {count} {target_class}(s) "
                f"were detected."
            )

        return (
            f"No {target_class} was detected "
            f"by YOLO."
        )

    # --------------------------
    # 2. Object Counting
    # --------------------------

    if task_type == "COUNT":

        if target_class is None:
            raise ValueError(
                "COUNT requires target_class"
            )

        count = counts.get(target_class, 0)

        return (
            f"YOLO detected {count} "
            f"{target_class}(s)."
        )

    # --------------------------
    # 3. Object Listing
    # --------------------------

    if task_type == "LIST":

        if not counts:
            return (
                "YOLO did not detect any "
                "supported objects."
            )

        object_list = [
            f"{class_name}: {count}"
            for class_name, count in sorted(
                counts.items()
            )
        ]

        return (
            "YOLO detected the following objects: "
            + ", ".join(object_list)
            + "."
        )

    raise ValueError(
        f"Unsupported detection task: {task_type}"
    )


@_profile_answer
def answer_missing_target(decision, detections):
    """Explain that a task cannot be answered when YOLO found no target."""

    target_class = decision.get("target_class")
    if target_class is None:
        raise ValueError(
            "Missing-target response requires target_class"
        )

    if any(
        detection["class"].lower() == target_class
        for detection in detections
    ):
        return None

    if decision.get("semantic_type") == "ocr":
        return (
            f"YOLO did not detect any {target_class}, "
            "so its text cannot be read."
        )

    attribute = decision.get("attribute") or "requested attribute"
    return (
        f"YOLO did not detect any {target_class}, "
        f"so its {attribute} cannot be determined."
    )
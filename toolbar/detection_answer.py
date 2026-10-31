
from collections import Counter


# ==============================
# Detection Answer
# ==============================

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

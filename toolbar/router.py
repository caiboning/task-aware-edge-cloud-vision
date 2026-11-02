
import re

# ==============================
# YOLO COCO Classes
# ==============================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# ==============================
# Class Aliases
# ==============================

CLASS_ALIASES = {
    "person": [
        "people", "man", "men", "woman", "women",
        "boy", "girl", "human", "pedestrian"
    ],
    "car": ["cars", "automobile", "automobiles"],
    "bus": ["buses"],
    "truck": ["trucks", "lorry", "lorries"],
    "bicycle": ["bicycles", "bike", "bikes"],
    "motorcycle": ["motorcycles", "motorbike"],
    "dog": ["dogs", "puppy", "puppies"],
    "cat": ["cats", "kitten", "kittens"],
    "zebra": ["zebras"],
    "horse": ["horses"],
    "bird": ["birds"],
    "chair": ["chairs"],
    "bottle": ["bottles"],
    "laptop": ["laptops"],
    "cell phone": ["phone", "phones", "mobile phone"],
    "tv": ["television", "televisions"]
}


# ==============================
# Text Utilities
# ==============================

def normalize_question(question):
    return re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()


def contains_phrase(text, phrase):
    pattern = (
        r"(?<!\w)"
        + re.escape(phrase)
        + r"(?!\w)"
    )
    return re.search(pattern, text) is not None


# ==============================
# Target Recognition
# ==============================

def find_target_class(question, detections=None):
    """
    Identify a COCO target class from
    the target description produced by Qwen.

    Returns:
        class name if exactly one class is matched
        None otherwise
    """

    q = normalize_question(question)
    matches = set()

    for class_name in COCO_CLASSES:

        aliases = [class_name]
        aliases.extend(CLASS_ALIASES.get(class_name, []))

        for alias in aliases:

            if contains_phrase(q, alias):
                matches.add(class_name)
                break

    if len(matches) == 1:
        return next(iter(matches))

    return None


# ==============================
# Semantic Task Router
# ==============================

def route_task(task, detections=None):
    """
    Map one Qwen task to the existing
    pipeline route names.

    Input:
        {
            "task_type": str,
            "target": str | None,
            "attribute": str | None
        }

    Returns:
        {
            "route": str,
            "task_type": str,
            "target_class": str | None,
            "target": str | None,
            "attribute": str | None,
            "semantic_type": str
        }
    """

    if not isinstance(task, dict):
        raise TypeError(
            "route_task expects a Qwen task dictionary"
        )

    kind = task.get("task_type")
    target = task.get("target")
    attribute = task.get("attribute")

    target_class = find_target_class(
        str(target or "")
    )

    if kind in ("existence", "counting", "listing"):

        general = (
            target is None
            or str(target).lower().strip() in {
                "image",
                "picture",
                "scene",
                "object",
                "objects",
                "item",
                "items",
                "all objects",
                "everything"
            }
        )

        if kind == "listing" and general:
            route = "DETECTIONS"
            task_type = "LIST"

        elif target_class is None:
            route = "UNSUPPORTED"
            task_type = kind.upper()

        else:
            route = "DETECTIONS"
            task_type = {
                "existence": "EXISTENCE",
                "counting": "COUNT",
                "listing": "LIST"
            }[kind]

    elif kind == "attribute":

        if target_class:
            route = "CROP"
        else:
            route = "FULL_IMAGE"

        task_type = "ATTRIBUTE"

    elif kind == "ocr":

        route = "FULL_IMAGE"
        task_type = "TEXT"

    elif kind == "complex":

        route = "FULL_IMAGE"
        task_type = "COMPLEX"

    elif kind == "cloud":

        route = "CLOUD"
        task_type = "CLOUD"

    else:

        route = "UNSUPPORTED"
        task_type = "UNKNOWN"

    return {
        "route": route,
        "task_type": task_type,
        "target_class": target_class,
        "target": target,
        "attribute": attribute,
        "semantic_type": kind
    }

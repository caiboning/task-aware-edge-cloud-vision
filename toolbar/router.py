
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
    Identify target classes from the question,
    independently of YOLO detections.

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
# Task Router
# ==============================

def route_task(question, detections=None):
    """
    Rule-based task classification.

    Returns:
        {
            "route": str,
            "task_type": str,
            "target_class": str | None
        }
    """

    q = normalize_question(question)

    target_class = find_target_class(question)

    # --------------------------
    # 1. Scene-level Questions
    # --------------------------

    scene_keywords = [
        "main color",
        "dominant color",
        "overall color",
        "main colour",
        "dominant colour",
        "overall colour",
        "describe the scene",
        "describe the image",
        "describe the picture",
        "what is happening",
        "what's happening",
        "whole image",
        "entire image",
        "whole picture"
    ]

    if any(k in q for k in scene_keywords):
        return {
            "route": "FULL_IMAGE",
            "task_type": "SCENE",
            "target_class": None
        }

    # --------------------------
    # 2. Text Recognition
    # --------------------------

    text_keywords = [
        "what text",
        "read the text",
        "what words",
        "what is written",
        "visible text",
        "read the sign",
        "what does the sign say"
    ]

    if any(k in q for k in text_keywords):
        return {
            "route": "FULL_IMAGE",
            "task_type": "TEXT",
            "target_class": None
        }

    # --------------------------
    # 3. Object Existence
    # --------------------------

    existence_patterns = [
        r"\bis there\b",
        r"\bare there\b",
        r"\bdo you see\b",
        r"\bcan you see\b",
        r"\bdoes .* contain\b",
        r"\bdoes .* have\b",
        r"\bis .* present\b",
        r"\bare .* present\b"
    ]

    if (
        target_class is not None
        and any(
            re.search(pattern, q)
            for pattern in existence_patterns
        )
    ):
        return {
            "route": "DETECTIONS",
            "task_type": "EXISTENCE",
            "target_class": target_class
        }

    # --------------------------
    # 4. Object Counting
    # --------------------------

    count_keywords = [
        "how many",
        "count",
        "number of"
    ]

    if (
        target_class is not None
        and any(k in q for k in count_keywords)
    ):
        return {
            "route": "DETECTIONS",
            "task_type": "COUNT",
            "target_class": target_class
        }

    # --------------------------
    # 5. Object Listing
    # --------------------------

    list_keywords = [
        "what objects",
        "which objects",
        "list the objects",
        "what items",
        "which items",
        "what can you see",
        "what is in the image",
        "what is in this picture"
    ]

    if any(k in q for k in list_keywords):
        return {
            "route": "DETECTIONS",
            "task_type": "LIST",
            "target_class": None
        }

    # --------------------------
    # 6. Object Attributes
    # --------------------------

    attribute_keywords = [
        "color",
        "colour",
        "wearing",
        "clothes",
        "clothing",
        "shirt",
        "jacket",
        "appearance",
        "look like",
        "shape",
        "pattern"
    ]

    if (
        target_class is not None
        and any(k in q for k in attribute_keywords)
    ):
        return {
            "route": "CROP",
            "task_type": "ATTRIBUTE",
            "target_class": target_class
        }

    # --------------------------
    # 7. Default
    # --------------------------

    return {
        "route": "LOCAL_ANSWER",
        "task_type": "GENERAL",
        "target_class": target_class
    }

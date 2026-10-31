
from collections import defaultdict

from toolbar import config


# ==============================
# Bounding Box Utilities
# ==============================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def count_connected_components(boxes, threshold):
    """
    Treat boxes as graph nodes.

    Two nodes are connected when their
    original bounding boxes have IoU >= threshold.

    Return the connected components.
    """

    n = len(boxes)
    visited = set()
    components = []

    for start in range(n):

        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component = []

        while stack:

            current = stack.pop()
            component.append(current)

            for other in range(n):

                if other in visited:
                    continue

                if calculate_iou(
                    boxes[current],
                    boxes[other]
                ) >= threshold:

                    visited.add(other)
                    stack.append(other)

        components.append(component)

    return components


def merge_boxes(boxes):
    """
    Return the smallest rectangle
    enclosing all provided boxes.
    """

    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes)
    ]


def crop_with_padding(image, bbox):
    """
    Crop image with configurable padding.
    """

    width, height = image.size

    x1, y1, x2, y2 = bbox

    padding_x = (
        x2 - x1
    ) * config.CROP_PADDING_RATIO

    padding_y = (
        y2 - y1
    ) * config.CROP_PADDING_RATIO

    x1 = max(0, int(x1 - padding_x))
    y1 = max(0, int(y1 - padding_y))
    x2 = min(width, int(x2 + padding_x + 0.9999))
    y2 = min(height, int(y2 + padding_y + 0.9999))

    if x2 <= x1 or y2 <= y1:
        return None, None

    final_bbox = [x1, y1, x2, y2]

    return image.crop(tuple(final_bbox)), final_bbox


# ==============================
# Crop Generation
# ==============================

def generate_crops(image, detections):
    """
    Generate eligible crops by YOLO class.

    Rules:
        1. Group detections by class.
        2. One box -> SINGLE crop.
        3. Multiple boxes forming one
           connected component -> MERGED crop.
        4. Multiple disconnected components
           -> FALLBACK, no crop.

    Returns:
        crops: dict[class_name, PIL.Image]
        crop_info: dict[class_name, metadata]
    """

    groups = defaultdict(list)

    for detection in detections:

        class_name = detection["class"].lower()
        bbox = detection["bbox"]

        groups[class_name].append(bbox)

    crops = {}
    crop_info = {}

    for class_name, boxes in groups.items():

        components = count_connected_components(
            boxes,
            config.CROP_IOU_THRESHOLD
        )

        if len(components) != 1:

            crop_info[class_name] = {
                "status": "FALLBACK",
                "reason": "MULTIPLE_DISCONNECTED_REGIONS",
                "matched_objects": len(boxes),
                "connected_components": len(components),
                "bbox": None
            }

            continue

        merged_bbox = merge_boxes(boxes)

        cropped, final_bbox = crop_with_padding(
            image,
            merged_bbox
        )

        if cropped is None:

            crop_info[class_name] = {
                "status": "FALLBACK",
                "reason": "INVALID_BBOX",
                "matched_objects": len(boxes),
                "connected_components": 1,
                "bbox": None
            }

            continue

        method = (
            "SINGLE"
            if len(boxes) == 1
            else "MERGED"
        )

        crops[class_name] = cropped

        crop_info[class_name] = {
            "status": "SUCCESS",
            "method": method,
            "matched_objects": len(boxes),
            "connected_components": 1,
            "bbox": final_bbox
        }

    return crops, crop_info

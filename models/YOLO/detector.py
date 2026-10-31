
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from toolbar import config


# ==============================
# YOLO Detector
# ==============================

class YOLODetector:
    """
    Local YOLO11 object detector.

    Supports:
        - Image path
        - PIL Image

    Returns:
        detections
        latency
    """

    def __init__(
        self,
        model_path=None,
        device=None,
        confidence=None
    ):

        self.model_path = (
            model_path
            if model_path is not None
            else config.YOLO_MODEL
        )

        self.device = (
            device
            if device is not None
            else config.YOLO_DEVICE
        )

        self.confidence = (
            confidence
            if confidence is not None
            else config.YOLO_CONF
        )

        # Load YOLO once
        self.model = YOLO(str(self.model_path))

    # ==============================
    # Object Detection
    # ==============================

    def detect(self, image, return_result=False):
        """
        Run YOLO inference.

        Args:
            image: str, Path, or PIL Image
            return_result: Return raw Ultralytics result

        Returns:
            detections: list
            latency: float (seconds)

            If return_result=True:
                detections, latency, result
        """

        if isinstance(image, (str, Path)):
            source = str(image)

        elif isinstance(image, Image.Image):
            source = image

        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        start = time.perf_counter()

        result = self.model.predict(
            source=source,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )[0]

        latency = time.perf_counter() - start

        detections = []

        for box in result.boxes:

            class_id = int(box.cls.item())

            detections.append({
                "class": result.names[class_id],
                "confidence": round(
                    float(box.conf.item()), 4
                ),
                "bbox": [
                    round(float(x), 2)
                    for x in box.xyxy[0].tolist()
                ]
            })

        if return_result:
            return detections, latency, result

        return detections, latency


from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """
    Local YOLO11 object detector.

    Input:
        Image path

    Output:
        List of detected objects
    """

    def __init__(self, model_path=None, device="cpu", confidence=0.25):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent
                / "yolo11n.pt"
            )

        self.model = YOLO(str(model_path))
        self.device = device
        self.confidence = confidence

    def detect(self, image_path):
        """
        Run object detection on an image.
        """

        results = self.model.predict(source=str(image_path), conf=self.confidence, device=self.device, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])

                detection = {
                    "class": self.model.names[class_id],
                    "confidence": round(
                        float(box.conf[0]), 4
                    ),
                    "bbox": [
                        round(float(x), 2)
                        for x in box.xyxy[0].tolist()
                    ]
                }

                detections.append(detection)

        return detections

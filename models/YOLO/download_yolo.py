from ultralytics import YOLO
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "yolo11n.pt"

# Download model if not available
if not MODEL_PATH.exists():
    model = YOLO("yolo11n.pt")
    model.save(str(MODEL_PATH))

# Load local model
model = YOLO(str(MODEL_PATH))

print("YOLO11n model is ready!")
print(f"Model path: {MODEL_PATH}")

from pathlib import Path

# ==============================
# Project Paths
# ==============================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

# ==============================
# Pipeline Configuration
# ==============================

IMAGE_DIR = DATA_DIR / "images_test"

PIPELINE_OUTPUT_DIR = OUTPUT_DIR / "images_test_result"

PIPELINE_CROP_DIR = PIPELINE_OUTPUT_DIR / "crops"
PIPELINE_YOLO_DIR = PIPELINE_OUTPUT_DIR / "yolo_detections"

PIPELINE_RESULTS = PIPELINE_OUTPUT_DIR / "results.json"
PIPELINE_SUMMARY = PIPELINE_OUTPUT_DIR / "summary.json"
PIPELINE_ERRORS = PIPELINE_OUTPUT_DIR / "errors.json"
PIPELINE_LOG = PIPELINE_OUTPUT_DIR / "pipeline.log"

# ==============================
# YOLO Configuration
# ==============================

YOLO_MODEL = (
    ROOT / "models" / "YOLO" / "yolo11n_ncnn_model"
)

YOLO_DEVICE = "cpu"
YOLO_CONF = 0.25

# ==============================
# YOLO Benchmark
# ==============================

COCO128_DIR = (
    DATA_DIR / "coco128" / "images" / "train2017"
)

YOLO_OUTPUT_DIR = OUTPUT_DIR / "coco128"
YOLO_IMAGE_OUTPUT_DIR = YOLO_OUTPUT_DIR / "images"

# ==============================
# VLM Configuration
# ==============================

VLM_MODEL = "ahmadwaqar/smolvlm2-256m-video:q8_0"

OLLAMA_HOST = "http://localhost:11434"

VLM_TEMPERATURE = 0
VLM_NUM_PREDICT = 128
VLM_NUM_CTX = 2048
VLM_REPEAT_PENALTY = 1.15

# ==============================
# Image Processing
# ==============================

IMAGE_QUALITY = 85

# ==============================
# Crop Configuration
# ==============================

CROP_IOU_THRESHOLD = 0.30
CROP_PADDING_RATIO = 0.05

# Only boxes of the same YOLO class
# are considered for merging.
CROP_SAME_CLASS_ONLY = True

# ==============================
# Benchmark Configuration
# ==============================

BENCHMARK_OUTPUT = (
    OUTPUT_DIR / "benchmark_results.json"
)

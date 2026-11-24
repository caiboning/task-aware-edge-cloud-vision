
import shutil
import tempfile
from ultralytics import YOLO
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent

MODEL_NAMES = [
    "yolo11n",
    "yolo11s",
    "yolov8n"
]


def download_yolo(model_name):

    PT_PATH = MODEL_DIR / f"{model_name}.pt"
    NCNN_PATH = MODEL_DIR / f"{model_name}_ncnn_model"

    NCNN_PARAM = NCNN_PATH / "model.ncnn.param"
    NCNN_BIN = NCNN_PATH / "model.ncnn.bin"
    NCNN_METADATA = NCNN_PATH / "metadata.yaml"

    required_files = (
        NCNN_PARAM,
        NCNN_BIN,
        NCNN_METADATA
    )

    # Skip download if the NCNN model is already complete.
    if all(path.is_file() for path in required_files):
        print(f"NCNN model ready: {NCNN_PATH}")
        return

    # Download model if necessary.
    if not PT_PATH.exists():
        print(f"Downloading {model_name}...")
        model = YOLO(f"{model_name}.pt")
        model.save(str(PT_PATH))

    # Export to NCNN.
    print(f"Exporting {model_name} to NCNN...")

    with tempfile.TemporaryDirectory(
        prefix=f"{model_name}_ncnn_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        if not temp_dir.as_posix().isascii():
            raise RuntimeError(
                f"NCNN export requires an ASCII temporary path; "
                f"got {temp_dir}."
            )

        temp_pt_path = temp_dir / PT_PATH.name

        shutil.copy2(PT_PATH, temp_pt_path)

        model = YOLO(str(temp_pt_path))
        model.export(format="ncnn", imgsz=640)

        exported_path = (
            temp_dir / f"{model_name}_ncnn_model"
        )

        if not all(
            (exported_path / path.name).is_file()
            for path in required_files
        ):
            raise RuntimeError(
                f"NCNN export did not create the expected "
                f"model files in {exported_path}."
            )

        shutil.copytree(
            exported_path,
            NCNN_PATH,
            dirs_exist_ok=True
        )

    if not all(path.is_file() for path in required_files):
        raise RuntimeError(
            f"NCNN export did not create the expected "
            f"model files in {NCNN_PATH}."
        )

    print(f"NCNN model ready: {NCNN_PATH}")


for model_name in MODEL_NAMES:
    download_yolo(model_name)

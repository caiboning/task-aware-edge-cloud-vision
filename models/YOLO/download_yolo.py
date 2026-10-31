import shutil
import tempfile
from ultralytics import YOLO
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PT_PATH = MODEL_DIR / "yolo11n.pt"
NCNN_PATH = MODEL_DIR / "yolo11n_ncnn_model"
NCNN_PARAM = NCNN_PATH / "model.ncnn.param"
NCNN_BIN = NCNN_PATH / "model.ncnn.bin"
NCNN_METADATA = NCNN_PATH / "metadata.yaml"

# Download model if necessary
if not PT_PATH.exists():
    print("Downloading YOLO11n...")
    model = YOLO("yolo11n.pt")
    model.save(str(PT_PATH))

# Export if the model files are missing, including after an interrupted export.
if not all(path.is_file() for path in (NCNN_PARAM, NCNN_BIN, NCNN_METADATA)):
    print("Exporting YOLO11n to NCNN...")
    with tempfile.TemporaryDirectory(prefix="yolo11n_ncnn_") as temp_dir:
        temp_dir = Path(temp_dir)
        if not temp_dir.as_posix().isascii():
            raise RuntimeError(
                f"NCNN export requires an ASCII temporary path; got {temp_dir}."
            )

        temp_pt_path = temp_dir / PT_PATH.name
        shutil.copy2(PT_PATH, temp_pt_path)
        model = YOLO(str(temp_pt_path))
        model.export(format="ncnn", imgsz=640)

        exported_path = temp_dir / f"{PT_PATH.stem}_ncnn_model"
        if not all(
            (exported_path / path.name).is_file()
            for path in (NCNN_PARAM, NCNN_BIN, NCNN_METADATA)
        ):
            raise RuntimeError(
                f"NCNN export did not create the expected model files in {exported_path}."
            )

        shutil.copytree(exported_path, NCNN_PATH, dirs_exist_ok=True)

if not all(path.is_file() for path in (NCNN_PARAM, NCNN_BIN, NCNN_METADATA)):
    raise RuntimeError(
        f"NCNN export did not create the expected model files in {NCNN_PATH}."
    )

print(f"NCNN model ready: {NCNN_PATH}")
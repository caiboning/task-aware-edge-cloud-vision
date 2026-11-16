from pathlib import Path

from toolbar import config


# Edit these values before running the paired comparison.
QUESTION = "What objects are in the image?"
IMAGE_DIR = config.DATA_DIR / "images_test"
OUTPUT_DIR = config.OUTPUT_DIR / "images_test_result"
PIPELINE_REPORT_NAME = "results"
BASELINE_REPORT_NAME = "baseline"


def configure_paths():
    image_dir = Path(IMAGE_DIR).expanduser()
    output_dir = Path(OUTPUT_DIR).expanduser()

    if not image_dir.is_absolute():
        image_dir = config.ROOT / image_dir
    if not output_dir.is_absolute():
        output_dir = config.ROOT / output_dir

    config.IMAGE_DIR = image_dir
    config.PIPELINE_OUTPUT_DIR = output_dir
    config.PIPELINE_CROP_DIR = output_dir / "crops"
    config.PIPELINE_YOLO_DIR = output_dir / "yolo_detections"
    config.PIPELINE_RESULTS = output_dir / "results.json"
    config.PIPELINE_SUMMARY = output_dir / "summary.json"
    config.PIPELINE_ERRORS = output_dir / "errors.json"
    config.PIPELINE_TEXT = output_dir / "results.txt"
    config.PIPELINE_LOG = output_dir / "pipeline.log"


def main():
    configure_paths()

    from baseline import find_images, run_baseline
    from pipeline import run_batch as run_task_aware_pipeline
    from toolbar.logger import get_logger

    logger = get_logger()
    question = QUESTION.strip()
    if not question:
        logger.warning("Set QUESTION in main.py before running")
        return

    image_paths = find_images()
    if not image_paths:
        logger.warning(
            "No supported images found in %s",
            config.IMAGE_DIR
        )
        return

    print(f"Question: {question}")
    print(f"Image directory: {config.IMAGE_DIR}")
    print(f"Output directory: {config.PIPELINE_OUTPUT_DIR}")
    print(f"Found {len(image_paths)} images.")

    print("\n========== Task-aware pipeline ==========")
    run_task_aware_pipeline(
        question=question,
        image_paths=image_paths,
        report_name=PIPELINE_REPORT_NAME
    )

    print("\n========== Full-image DeepSeek baseline ==========")
    run_baseline(
        question,
        image_paths=image_paths,
        report_name=BASELINE_REPORT_NAME
    )


if __name__ == "__main__":
    main()

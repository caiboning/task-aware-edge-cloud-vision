import argparse
import json
import time

from datetime import datetime
from pathlib import Path

from models.DEEPSEEK.deepseek import DEEPSEEK_MODEL, ask_deepseek
from toolbar import config
from toolbar.logger import get_logger
from toolbar.profiler import collect_stage_profiles, cloud_body_sizes


logger = get_logger()


def get_report_path(report_name):
    """Create a unique timestamped baseline report path."""

    report_name = Path(report_name)
    if (
        report_name.name != str(report_name)
        or report_name.stem in {"", ".", ".."}
    ):
        raise ValueError(
            "Report name must be a filename or filename stem, "
            "not a path."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (
        config.PIPELINE_OUTPUT_DIR
        / f"{report_name.stem}_{timestamp}.txt"
    )

    suffix = 1
    while report_path.exists():
        report_path = (
            config.PIPELINE_OUTPUT_DIR
            / f"{report_name.stem}_{timestamp}_{suffix}.txt"
        )
        suffix += 1

    return report_path


def find_images():
    if not config.IMAGE_DIR.exists():
        return []

    return sorted(
        path for path in config.IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def write_report(results, summary, report_path):
    lines = [
        "Full-Image DeepSeek Baseline",
        "=" * 29,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Question: {summary['question']}",
        f"Model: {summary['model']}",
        "",
        "Batch Summary",
        "-" * 13,
        f"Images: {summary['total_images']}",
        f"Successful: {summary['successful_images']}",
        f"Failed: {summary['failed_images']}",
        f"Average end-to-end latency: {summary['average_end_to_end_latency_ms']} ms",
        f"Total estimated Edge-to-Cloud request bytes: {summary['total_request_bytes']}",
        f"Average estimated request bytes per image: {summary['average_request_bytes']} bytes",
        "Request size is serialized JSON including the base64 image and prompt; "
        "it excludes HTTP headers, TLS overhead, and response traffic.",
        "",
        "Per-Image Results",
        "-" * 17,
    ]

    for index, result in enumerate(results, start=1):
        lines.extend([
            "",
            f"[{index}/{len(results)}] {result['image']}",
            f"Status: {result['status']}",
        ])
        if result["status"] == "FAILED":
            lines.extend([
                f"Error: {result['error_type']}",
                f"Details: {result['error_message']}",
            ])
            continue

        lines.extend([
            "Route: FULL_IMAGE -> Cloud (DeepSeek)",
            "Image used: FULL_IMAGE",
            f"Answer: {result['answer']}",
            f"API latency: {result['api_latency_ms']} ms",
            f"End-to-end latency: {result['end_to_end_latency_ms']} ms",
            f"Estimated Edge-to-Cloud request body: {result['request_bytes']} bytes",
        ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def run_baseline(question, image_paths=None, report_name="baseline"):
    """Run full-image DeepSeek on each image using the same question."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if image_paths is None:
        image_paths = find_images()
    else:
        image_paths = list(image_paths)

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {config.IMAGE_DIR}"
        )

    report_path = get_report_path(report_name)
    config.PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    batch_start = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path).resolve()
        image_name = str(
            image_path.relative_to(config.IMAGE_DIR.resolve())
        )
        print(
            f"\n[Baseline {index}/{len(image_paths)}] "
            f"Processing: {image_name}"
        )

        start = time.perf_counter()
        try:
            with collect_stage_profiles() as stages:
                answer, api_latency, request_bytes = ask_deepseek(
                    image_path,
                    question,
                    return_request_bytes=True
                )

            actual_request, actual_response = cloud_body_sizes(stages, 1)
            actual_total = (
                actual_request + actual_response
                if actual_request is not None and actual_response is not None
                else None
            )

            result = {
                "image": image_name,
                "question": question,
                "status": "SUCCESS",
                "route": "CLOUD_ANSWER",
                "used_image": "FULL_IMAGE",
                "answer": answer,
                "api_latency_ms": round(api_latency * 1000, 2),
                "end_to_end_latency_ms": round(
                    (time.perf_counter() - start) * 1000,
                    2
                ),
                "request_bytes": request_bytes,
                "runtime_metrics": stages,
                "communication_metrics": {
                    "request_bytes": actual_request,
                    "response_bytes": actual_response
                },
                "edge_cloud_request_bytes": actual_request,
                "edge_cloud_response_bytes": actual_response,
                "edge_cloud_total_bytes": actual_total
            }
            results.append(result)
            print("Answer:", answer)
            print("API latency:", result["api_latency_ms"], "ms")
            print("Estimated Edge-to-Cloud payload:", request_bytes, "bytes")
        except Exception as error:
            error_result = {
                "image": image_name,
                "question": question,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
            results.append(error_result)
            errors.append(error_result)
            logger.exception(
                "DeepSeek baseline failed | Image: %s",
                image_name
            )

    successful_results = [
        result for result in results
        if result["status"] == "SUCCESS"
    ]
    successful_count = len(successful_results)
    total_request_bytes = sum(
        result["request_bytes"]
        for result in successful_results
    )
    summary = {
        "question": question,
        "model": DEEPSEEK_MODEL,
        "total_images": len(image_paths),
        "successful_images": successful_count,
        "failed_images": len(errors),
        "average_api_latency_ms": round(
            sum(result["api_latency_ms"] for result in successful_results)
            / successful_count,
            2
        ) if successful_count else 0,
        "average_end_to_end_latency_ms": round(
            sum(
                result["end_to_end_latency_ms"]
                for result in successful_results
            ) / successful_count,
            2
        ) if successful_count else 0,
        "total_request_bytes": total_request_bytes,
        "average_request_bytes": round(
            total_request_bytes / successful_count,
            2
        ) if successful_count else 0,
        "batch_latency_s": round(
            time.perf_counter() - batch_start,
            3
        )
    }

    output_dir = config.PIPELINE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_results.json").write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    (output_dir / "baseline_errors.json").write_text(
        json.dumps(errors, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    write_report(results, summary, report_path)

    print("\n========== DeepSeek Baseline Summary ==========")
    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print("Readable baseline report:", report_path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full-image DeepSeek baseline."
    )
    parser.add_argument(
        "--question",
        help="Question to ask for every image. Prompts if omitted."
    )
    parser.add_argument(
        "--report-name",
        default="baseline",
        help="Baseline report filename or stem."
    )
    args = parser.parse_args(argv)

    question = args.question
    if question is None:
        try:
            question = input("\nBaseline question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting baseline.")
            return

    if not question:
        raise ValueError("Question cannot be empty.")

    run_baseline(question, report_name=args.report_name)


if __name__ == "__main__":
    main()

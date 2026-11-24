"""Measure payload byte counts and store model profiles."""

import json
import logging
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

from toolbar import config


_PROFILE_LOCK = Lock()
_PIPELINE_STAGES = ContextVar("pipeline_profile_stages", default=None)


def file_size_bytes(path: str | Path) -> int:
    """Return the on-disk length of a regular file in bytes."""
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    return info.st_size


def json_size_bytes(data: object) -> int:
    """Return the size of compact UTF-8 JSON."""
    return len(
        json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )


def payload_size_bytes(data: object) -> int | None:
    """Return logical payload bytes, or None for an unavailable payload."""
    if data is None:
        return None
    if isinstance(data, Path):
        return file_size_bytes(data)
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, memoryview):
        return data.nbytes

    # PyTorch-compatible tensors and NumPy arrays need no serialization.
    numel = getattr(data, "numel", None)
    element_size = getattr(data, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    nbytes = getattr(data, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    return json_size_bytes(data)


def update_model_profiles(update):
    """Update the shared JSON while preserving other sections."""
    output = config.OUTPUT_DIR / "model_profiles.json"
    with _PROFILE_LOCK:
        try:
            contents = json.loads(output.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            contents = {"models": {}}

        update(contents)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(contents, temporary, ensure_ascii=False, indent=4)
                temporary.write("\n")

            temporary_path.replace(output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _append_profile(collection, profile):
    try:
        update_model_profiles(
            lambda contents: contents.setdefault(collection, []).append(profile)
        )
    except (OSError, ValueError, AttributeError, TypeError) as error:
        logging.getLogger("pipeline").warning(
            "Could not save %s: %s", collection, error
        )


def save_stage_profile(profile, collection="stage_profiles"):
    stages = _PIPELINE_STAGES.get()
    if stages is None:
        _append_profile(collection, profile)
    else:
        stages.append(profile)


def _sum_bytes(stages, field):
    values = [stage.get(field) for stage in stages]
    return sum(values) if all(value is not None for value in values) else None


@contextmanager
def collect_stage_profiles():
    stages = []
    token = _PIPELINE_STAGES.set(stages)
    try:
        yield stages
    finally:
        _PIPELINE_STAGES.reset(token)


def cloud_body_sizes(stages, expected_calls):
    cloud = [s for s in stages if s["stage"] == "deepseek_cloud_inference"]
    if len(cloud) != expected_calls:
        return None, None

    return (
        _sum_bytes(cloud, "network_request_bytes"),
        _sum_bytes(cloud, "network_response_bytes")
    )


def profile_pipeline(function):
    @wraps(function)
    def wrapped(image_path, question):
        with collect_stage_profiles() as stages:
            result = function(image_path, question)

        result["runtime_metrics"] = stages

        metrics = result["metrics"]
        local = [s for s in stages if s["stage"] == "smolvlm_inference"]
        expected_cloud_calls = sum(
            task["route"] in {"CROP", "CLOUD_ANSWER"}
            for task in result["task_results"]
        )
        expected_local_calls = sum(
            task["route"] == "FULL_IMAGE" for task in result["task_results"]
        )

        request_bytes, response_bytes = cloud_body_sizes(
            stages, expected_cloud_calls
        )
        total_bytes = (
            request_bytes + response_bytes
            if request_bytes is not None and response_bytes is not None
            else None
        )

        metrics["edge_cloud_request_bytes"] = request_bytes
        metrics["edge_cloud_response_bytes"] = response_bytes
        metrics["edge_cloud_total_bytes"] = total_bytes

        result["communication_metrics"] = {
            "request_bytes": request_bytes,
            "response_bytes": response_bytes
        }

        local_summary = {
            key: _sum_bytes(local, field) if len(local) == expected_local_calls else None
            for key, field in (
                ("smolvlm_logical_input_bytes", "input_payload_bytes"),
                ("local_api_request_bytes", "local_api_request_bytes"),
                ("local_api_response_bytes", "local_api_response_bytes"),
            )
        }

        _append_profile("pipeline_runs", {
            "image": result["image"],
            "question": result["question"],
            "route": result["route"],
            "requested_route": result["requested_route"],
            "used_image": result["used_image"],
            "status": result["status"],
            "stages": stages,
            "summary": {
                "total_latency_ms": metrics["total_latency_ms"],
                "qwen_model_loading_ms": metrics["llm_model_load_latency_ms"],
                "yolo_constructor_ms": metrics["yolo_model_load_latency_ms"],
                "estimated_edge_cloud_request_bytes": metrics["edge_to_cloud_bytes"],
                "edge_cloud_request_bytes": request_bytes,
                "edge_cloud_response_bytes": response_bytes,
                "edge_cloud_total_bytes": total_bytes,
                **local_summary,
            }
        })
        return result

    return wrapped
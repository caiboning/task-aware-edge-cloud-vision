"""Static model storage and metadata collection."""

import json
import os
import re
from pathlib import Path
from gguf import GGUFReader
from gguf import LlamaFileType

from toolbar import config
from toolbar.profiler import file_size_bytes, update_model_profiles


def new_profile(name, model_format=None):
    return {
        "model_name": name,
        "model_format": model_format,
        "weight_size_bytes": None,
        "total_model_size_bytes": None,
        "parameter_count": None,
        "precision": None,
        "flops": None,
        "input_resolution": None,
        "fixed_prompt_bytes": None,
    }


def available_size(path):
    try:
        return file_size_bytes(path)
    except (OSError, ValueError):
        return None


def ncnn_profile(name, directory):
    profile = new_profile(name, "NCNN")

    weight = available_size(directory / "model.ncnn.bin")
    structure = available_size(directory / "model.ncnn.param")
    metadata_path = directory / "metadata.yaml"
    metadata = available_size(metadata_path)

    profile["weight_size_bytes"] = weight

    if weight is not None and structure is not None:
        if metadata is not None or not metadata_path.exists():
            profile["total_model_size_bytes"] = (
                weight + structure + (metadata or 0)
            )

    try:
        import yaml

        metadata = yaml.safe_load(
            metadata_path.read_text(encoding="utf-8")
        ) or {}

        if not isinstance(metadata, dict):
            return profile

        profile["input_resolution"] = metadata.get("imgsz")
        args = metadata.get("args") or {}
        profile["precision"] = args.get("quantize")

    except (ImportError, OSError, ValueError, TypeError):
        pass

    return profile


def qwen_profile(path):
    profile = new_profile("Qwen", "GGUF")

    size = available_size(path)

    profile["weight_size_bytes"] = size
    profile["total_model_size_bytes"] = size

    if size is not None:
        try:
            from gguf import GGUFReader

            reader = GGUFReader(str(path))

            profile["parameter_count"] = sum(
                int(t.n_elements) for t in reader.tensors
            )

            profile["parameter_count_basis"] = (
                "stored GGUF tensor elements"
            )

            field = reader.fields.get("general.file_type")

            if field is not None:
                from gguf import LlamaFileType

                profile["precision"] = LlamaFileType(
                    int(field.parts[field.data[0]][0])
                ).name

        except (
            ImportError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError
        ):
            pass

    return profile


def manifest_path(store, model):
    name, separator, tag = model.rpartition(":")

    if not separator or "/" in tag:
        name, tag = model, "latest"

    parts = name.split("/")
    registry = "registry.ollama.ai"

    if len(parts) > 1 and (
        "." in parts[0]
        or ":" in parts[0]
        or parts[0] == "localhost"
    ):
        registry = parts.pop(0)

    if len(parts) == 1:
        parts.insert(0, "library")

    if any(
        part in ("", ".", "..")
        for part in [registry, *parts, tag]
    ):
        raise ValueError(f"Invalid Ollama model name: {model}")

    return store.joinpath(
        "manifests", registry, *parts, tag
    )


def read_json_object(path):
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None

    return data if isinstance(data, dict) else None


def ollama_profile(store, model):
    profile = new_profile("SmolVLM")

    manifest = manifest_path(store, model)
    manifest_bytes = available_size(manifest)
    data = read_json_object(manifest)

    if manifest_bytes is None or data is None:
        return profile

    layers = data.get("layers")
    settings = data.get("config")

    if not isinstance(layers, list) or not isinstance(settings, dict):
        return profile

    blobs = {}
    weights = set()
    model_weights = set()
    metadata = {}

    for entry in [settings, *layers]:

        if not isinstance(entry, dict):
            return profile

        digest = entry.get("digest")

        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", digest
        ):
            return profile

        blob = store / "blobs" / digest.replace(":", "-")

        if digest not in blobs:
            blobs[digest] = available_size(blob)

        if entry is settings:
            if blobs[digest] is not None:
                metadata = read_json_object(blob) or {}
            continue

        media_type = entry.get("mediaType")

        if media_type == "application/vnd.ollama.image.model":
            model_weights.add(digest)

        if media_type in {
            "application/vnd.ollama.image.model",
            "application/vnd.ollama.image.projector",
            "application/vnd.ollama.image.adapter",
        }:
            weights.add(digest)

    if model_weights and all(
        blobs[digest] is not None for digest in weights
    ):
        profile["weight_size_bytes"] = sum(
            blobs[digest] for digest in weights
        )

    if model_weights and all(
        size is not None for size in blobs.values()
    ):
        profile["total_model_size_bytes"] = (
            manifest_bytes + sum(blobs.values())
        )

    model_format = metadata.get("model_format")
    precision = metadata.get("file_type")

    if isinstance(model_format, str) and model_format:
        profile["model_format"] = model_format.upper()

    if (
        isinstance(precision, str)
        and precision
        and weights == model_weights
    ):
        profile["precision"] = precision

    return profile


def collect_profiles(ollama_store):
    yolo_dir = config.YOLO_MODEL.parent

    models = {
        "YOLO11n": ncnn_profile(
            "YOLO11n",
            yolo_dir / "yolo11n_ncnn_model"
        ),

        "YOLO11s": ncnn_profile(
            "YOLO11s",
            yolo_dir / "yolo11s_ncnn_model"
        ),

        "YOLOv8n": ncnn_profile(
            "YOLOv8n",
            yolo_dir / "yolov8n_ncnn_model"
        ),

        "Qwen": qwen_profile(
            config.ROOT / "models" / "Qwen" / "Qwen3-0.6B-Q4_0.gguf"
        ),

        "SmolVLM": ollama_profile(
            ollama_store,
            config.VLM_MODEL
        ),
    }

    return {"models": models}


def refresh_profiles():
    store = Path(
        os.environ.get("OLLAMA_MODELS")
        or Path.home() / ".ollama" / "models"
    )

    profiles = collect_profiles(store)

    update_model_profiles(
        lambda data: data.update(profiles)
    )

    return profiles["models"]
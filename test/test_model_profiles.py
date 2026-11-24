"""Collect static model profiles without loading the models."""

import argparse
import json
import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from toolbar import config
from toolbar.profiler import (
    file_size_bytes,
    json_size_bytes,
    payload_size_bytes,
    update_model_profiles,
)


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
            profile["total_model_size_bytes"] = weight + structure + (metadata or 0)
    return profile


def qwen_profile(path):
    profile = new_profile("Qwen", "GGUF")
    size = available_size(path)
    profile["weight_size_bytes"] = size
    profile["total_model_size_bytes"] = size
    return profile


def manifest_path(store, model):
    name, separator, tag = model.rpartition(":")
    if not separator or "/" in tag:
        name, tag = model, "latest"

    parts = name.split("/")
    registry = "registry.ollama.ai"
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry = parts.pop(0)
    if len(parts) == 1:
        parts.insert(0, "library")

    if any(part in ("", ".", "..") for part in [registry, *parts, tag]):
        raise ValueError(f"Invalid Ollama model name: {model}")
    return store.joinpath("manifests", registry, *parts, tag)


def read_json_object(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
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

    if model_weights and all(blobs[digest] is not None for digest in weights):
        profile["weight_size_bytes"] = sum(blobs[digest] for digest in weights)
    if model_weights and all(size is not None for size in blobs.values()):
        profile["total_model_size_bytes"] = manifest_bytes + sum(blobs.values())

    model_format = metadata.get("model_format")
    precision = metadata.get("file_type")
    if isinstance(model_format, str) and model_format:
        profile["model_format"] = model_format.upper()
    if isinstance(precision, str) and precision and weights == model_weights:
        profile["precision"] = precision
    return profile


def collect_profiles(ollama_store):
    yolo_dir = config.YOLO_MODEL.parent
    models = {
        "YOLO11n": ncnn_profile("YOLO11n", yolo_dir / "yolo11n_ncnn_model"),
        "YOLO11s": ncnn_profile("YOLO11s", yolo_dir / "yolo11s_ncnn_model"),
        "YOLOv8n": ncnn_profile("YOLOv8n", yolo_dir / "yolov8n_ncnn_model"),
        "Qwen": qwen_profile(config.ROOT / "models" / "Qwen" / "Qwen3-0.6B-Q4_0.gguf"),
        "SmolVLM": ollama_profile(ollama_store, config.VLM_MODEL),
    }
    return {"models": models}


class ModelProfileTests(unittest.TestCase):
    def test_payloads(self):
        class Tensor:
            def numel(self):
                return 3

            def element_size(self):
                return 4

        self.assertEqual(payload_size_bytes("猫"), 3)
        self.assertEqual(json_size_bytes({"猫": [1]}), 11)
        self.assertEqual(payload_size_bytes(b"abcd"), 4)
        self.assertEqual(payload_size_bytes(memoryview(b"abcd").cast("H")), 4)
        self.assertEqual(payload_size_bytes(Tensor()), 12)
        self.assertIsNone(payload_size_bytes(None))
        with TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            self.assertEqual(payload_size_bytes(image), 14)

    def test_ncnn_and_gguf_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.ncnn.bin").write_bytes(b"1234")
            self.assertIsNone(ncnn_profile("YOLO11n", root)["total_model_size_bytes"])
            (root / "model.ncnn.param").write_bytes(b"12")
            (root / "metadata.yaml").write_bytes(b"123")
            (root / "unrelated.bin").write_bytes(b"ignored")
            profile = ncnn_profile("YOLO11n", root)
            self.assertEqual(profile["weight_size_bytes"], 4)
            self.assertEqual(profile["total_model_size_bytes"], 9)

            gguf = root / "model.gguf"
            self.assertIsNone(qwen_profile(gguf)["weight_size_bytes"])
            gguf.write_bytes(b"12345")
            self.assertEqual(qwen_profile(gguf)["total_model_size_bytes"], 5)

    def test_ollama_counts_unique_blobs_for_selected_tag(self):
        with TemporaryDirectory() as temporary:
            store = Path(temporary)
            (store / "blobs").mkdir()
            settings = b'{"model_format":"gguf","file_type":"Q8_0"}'
            entries = []
            for key, data, media_type in (
                ("a", settings, "application/vnd.docker.container.image.v1+json"),
                ("b", b"weights", "application/vnd.ollama.image.model"),
                ("c", b"projector", "application/vnd.ollama.image.projector"),
                ("d", b"template", "application/vnd.ollama.image.template"),
            ):
                digest = key * 64
                (store / "blobs" / f"sha256-{digest}").write_bytes(data)
                entries.append({"digest": f"sha256:{digest}", "mediaType": media_type})

            model = "example/smolvlm:q8_0"
            manifest = store / "manifests/registry.ollama.ai/example/smolvlm/q8_0"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"config": entries[0], "layers": entries[1:] + [entries[1]]}),
                encoding="utf-8",
            )
            (store / "blobs" / ("sha256-" + "e" * 64)).write_bytes(b"unused")
            profile = ollama_profile(store, model)
            self.assertEqual(profile["model_format"], "GGUF")
            self.assertIsNone(profile["precision"])
            self.assertEqual(profile["weight_size_bytes"], 16)
            self.assertEqual(
                profile["total_model_size_bytes"],
                manifest.stat().st_size + len(settings) + 16 + len(b"template"),
            )
            self.assertIsNone(ollama_profile(store, "example/smolvlm")["weight_size_bytes"])

            (store / "blobs" / ("sha256-" + "d" * 64)).unlink()
            profile = ollama_profile(store, model)
            self.assertEqual(profile["weight_size_bytes"], 16)
            self.assertIsNone(profile["total_model_size_bytes"])

            (store / "blobs" / ("sha256-" + "c" * 64)).unlink()
            profile = ollama_profile(store, model)
            self.assertIsNone(profile["weight_size_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama-models",
        type=Path,
        default=Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models"),
        help="Model storage directory used by the Ollama service.",
    )
    args = parser.parse_args()

    profiles = collect_profiles(args.ollama_models)
    output = config.OUTPUT_DIR / "model_profiles.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profiles, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(output)
    for name, profile in profiles["models"].items():
        if profile["total_model_size_bytes"] is None:
            print(f"{name}: model storage is missing, incomplete, or unreadable")


if __name__ == "__main__":
    main()
"""Sample local model resources without running the cloud baseline."""

import argparse
import json
import time
from pathlib import Path
from threading import Event, Thread
from urllib.parse import urlsplit

import psutil
from PIL import Image

from toolbar import config
from toolbar.profiler import save_stage_profile


SAMPLE_INTERVAL_SECONDS = 0.02
YOLO_MODELS = {
    "YOLO11n": "yolo11n_ncnn_model",
    "YOLO11s": "yolo11s_ncnn_model",
    "YOLOv8n": "yolov8n_ncnn_model"
}


def measure(stage, model, operation, read_rss, scope, process, **fields):
    """Measure one operation and sample RSS while it runs."""
    stopped = Event()
    samples = []

    def sample():
        while not stopped.wait(SAMPLE_INTERVAL_SECONDS):
            value = read_rss()
            if value is not None:
                samples.append(value)

    before = read_rss()
    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        result = operation()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        stopped.set()
        after = read_rss()
        sampler.join()

    peak = None
    if samples:
        peak = max(value for value in [before, *samples, after] if value is not None)

    return result, {
        "stage": stage,
        "model": model,
        "latency_ms": round(elapsed_ms, 3),
        "memory_scope": scope,
        "memory_pid": process.pid if process is not None else None,
        "sample_interval_ms": SAMPLE_INTERVAL_SECONDS * 1000,
        "memory_sample_count": len(samples),
        "rss_before_bytes": before,
        "peak_ram_bytes": peak,
        "rss_after_bytes": after,
        **fields
    }


def python_rss(process):
    try:
        return process.memory_info().rss
    except psutil.Error:
        return None


def find_ollama_server():
    host = urlsplit(config.OLLAMA_HOST)
    if host.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None

    port = host.port or (443 if host.scheme == "https" else 80)
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None

    servers = {}
    for connection in connections:
        if (
            connection.status != psutil.CONN_LISTEN
            or connection.pid is None
            or connection.laddr.port != port
            or connection.laddr.ip not in {"127.0.0.1", "::1", "0.0.0.0", "::"}
        ):
            continue
        try:
            process = psutil.Process(connection.pid)
            if process.name().lower() in {"ollama", "ollama.exe"}:
                servers[process.pid] = process
        except psutil.Error:
            continue

    return next(iter(servers.values())) if len(servers) == 1 else None


def ollama_tree_rss(server, runner_seen):
    if server is None:
        return None
    try:
        children = server.children(recursive=True)
        values = [process.memory_info().rss for process in [server, *children]]
    except psutil.Error:
        return None

    for child in children:
        try:
            name = child.name().lower()
            if (
                name in {"llama-server", "llama-server.exe"}
                or "ollama_llama_server" in name
                or "runner" in child.cmdline()[1:]
            ):
                runner_seen.set()
        except psutil.Error:
            continue

    return sum(values)


def emit(profile):
    save_stage_profile(profile)
    print(json.dumps(profile, ensure_ascii=False, indent=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["Qwen", *YOLO_MODELS, "SmolVLM"])
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question", default="Is there a car?")
    args = parser.parse_args()

    if args.model != "Qwen" and (args.image is None or not args.image.is_file()):
        parser.error("--image must point to an existing image for YOLO or SmolVLM")

    if args.model == "SmolVLM":
        from ollama import Client
        from models.SmolVLM.vlm import ask_vlm

        class ResourceClient(Client):
            def chat(self, *args, **kwargs):
                response = super().chat(*args, **kwargs)
                self.load_duration_ns = getattr(response, "load_duration", None)
                return response

        server = find_ollama_server()
        runner_seen = Event()
        client = ResourceClient(host=config.OLLAMA_HOST)
        with Image.open(args.image) as source:
            image = source.convert("RGB")

        (_, api_seconds), profile = measure(
            "smolvlm_resources", config.VLM_MODEL,
            lambda: ask_vlm(image, args.question, client=client),
            lambda: ollama_tree_rss(server, runner_seen),
            "ollama_server_process_tree", server
        )
        load_ns = client.load_duration_ns
        profile["loading_latency_ms"] = round(load_ns / 1_000_000, 3) if load_ns is not None else None
        profile["api_latency_ms"] = round(api_seconds * 1000, 3)
        profile["runner_observed"] = runner_seen.is_set()
        if not runner_seen.is_set():
            profile["peak_ram_bytes"] = None
        emit(profile)
        return

    process = psutil.Process()

    def read_rss():
        return python_rss(process)

    if args.model == "Qwen":
        from models.Qwen.llm import LocalLLM

        llm, profile = measure(
            "qwen_model_loading", args.model, LocalLLM, read_rss,
            "python_process", process
        )
        profile["loading_latency_ms"] = profile["latency_ms"]
        emit(profile)
        _, profile = measure(
            "qwen_classification_resources", args.model,
            lambda: llm.understand_task(args.question), read_rss,
            "python_process", process
        )
        emit(profile)
        return

    from models.YOLO.detector import YOLODetector

    model_path = config.YOLO_MODEL.parent / YOLO_MODELS[args.model]
    detector, profile = measure(
        "yolo_constructor", args.model,
        lambda: YOLODetector(model_path=model_path), read_rss,
        "python_process", process
    )
    emit(profile)
    for phase in ("first_prediction", "subsequent_prediction"):
        (_, predict_seconds, result), profile = measure(
            "yolo_detection_resources", args.model,
            lambda: detector.detect(args.image, return_result=True),
            read_rss, "python_process", process, phase=phase
        )
        profile["predict_latency_ms"] = round(predict_seconds * 1000, 3)
        profile["inference_latency_ms"] = result.speed.get("inference")
        profile["loading_latency_ms"] = None
        emit(profile)


if __name__ == "__main__":
    main()
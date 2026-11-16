# Task-Aware Edge-to-Cloud Visual Communication for Physical AI

## Project Overview
This project explores task-aware semantic communication for Physical AI systems. Instead of always transmitting full-resolution images to the cloud, the system selects what visual information to process or transmit based on the user's query and available resources.

The system supports multiple strategies, including local inference using lightweight vision-language models (VLMs), transmitting object detection results, sending cropped image regions, or forwarding full images to a backend VLM.

Our goal is to reduce communication overhead and end-to-end latency while maintaining task accuracy. We evaluate different strategies based on task accuracy, transmitted bytes, and end-to-end latency.

## How to Use
Run both the task-aware pipeline and the full-image DeepSeek baseline on the same images and question:

```bash
python main.py
```

Before running, edit `QUESTION`, `IMAGE_DIR`, `OUTPUT_DIR`, `PIPELINE_REPORT_NAME`, and `BASELINE_REPORT_NAME` near the top of `main.py`. Both strategies use the same question and image directory.

The runs execute sequentially, not concurrently. Each text report gets a timestamp. Baseline JSON outputs (`baseline_results.json`, `baseline_summary.json`, and `baseline_errors.json`) are saved separately in `OUTPUT_DIR`. The baseline sends every full image and the original question to DeepSeek, so it makes one cloud request per image.

Run the task-aware pipeline through `python main.py`. To run only the full-image baseline independently, use `python baseline.py`.

The task-aware pipeline scans `IMAGE_DIR` (including subdirectories) and applies `QUESTION` from `main.py` to each image. Qwen classifies the request as existence, counting, listing, attribute, OCR, or complex. Object detection/listing and supported object counting use YOLO; supported-target attribute and OCR tasks use YOLO to detect and crop the target, then send the crop to DeepSeek. If the target is missing, YOLO reports that; if a crop cannot be generated, the full image is sent to DeepSeek as a fallback. OCR without a specific supported target uses SmolVLM with the full image. Other full-image visual tasks use SmolVLM, while complex questions, future predictions, unsupported targets, and requests that do not fit another category go to DeepSeek. YOLO is skipped when none of the tasks needs detections or crops.

Each run saves machine-readable results (`results.json`, `summary.json`, and `errors.json`) and a uniquely named readable report in `OUTPUT_DIR`. The JSON files are overwritten with the latest run; previous text reports are kept.

Latency reports separate image loading, Qwen model loading and inference, routing, YOLO model loading and inference, crop generation, SmolVLM preparation and calls, DeepSeek preparation and API calls, and other pipeline overhead. Summary component averages are per successful image; stages not used for an image count as 0. Model-load time is charged to the image that first triggers loading.

Per-image reports show the model(s) used for each route: `DETECTIONS` uses YOLO, targeted `CROP` uses YOLO and Cloud (DeepSeek), `FULL_IMAGE` uses SmolVLM, and `CLOUD_ANSWER` uses Cloud (DeepSeek). Reports identify whether the cloud received a crop or the full image and include the saved crop path when applicable. If YOLO finds no target for an attribute or targeted OCR question, the pipeline reports that the requested information cannot be determined instead of asking a model to guess.

Payload reports only estimated edge-to-cloud request-body bytes. Local Qwen and YOLO processing contributes 0 bytes; DeepSeek routes include the serialized JSON request with its prompt and base64-encoded full image or target crop. This estimate excludes HTTP headers, TLS overhead, and cloud-to-edge response traffic, so it is not a wire-level network measurement.

The Qwen classifier requires `llama-cpp-python` and the `Qwen3-0.6B-Q4_0.gguf` weights at `models/Qwen/Qwen3-0.6B-Q4_0.gguf`; model weights are not stored in Git. To use the DeepSeek route, set `DEEPSEEK_API_KEY` in `models/DEEPSEEK/.env`. Optionally set `DEEPSEEK_MODEL` there (default: `deepseek-flash`) and `DEEPSEEK_MAX_TOKENS` (default: `4096`). Keep API keys private and do not commit the `.env` file.

Qwen routes requests based on semantic intent rather than fixed English keywords. There is no separate `cloud` task category: `complex` is the catch-all category and uses DeepSeek.
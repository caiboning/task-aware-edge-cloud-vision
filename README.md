# Task-Aware Edge-to-Cloud Visual Communication for Physical AI

## Project Overview
This project explores task-aware semantic communication for Physical AI systems. Instead of always transmitting full-resolution images to the cloud, the system selects what visual information to process or transmit based on the user's query and available resources.

The system supports multiple strategies, including local inference using lightweight vision-language models (VLMs), transmitting object detection results, sending cropped image regions, or forwarding full images to a backend VLM.

Our goal is to reduce communication overhead and end-to-end latency while maintaining task accuracy. We evaluate different strategies based on task accuracy, transmitted bytes, and end-to-end latency.

## How to Use
Run the batch pipeline with:

```bash
python pipeline.py
```

You can give the readable report a custom name from the terminal. Each run adds a timestamp and keeps earlier text reports:

```bash
python pipeline.py --report-name zebra_test
```

Reports are saved under `outputs/images_test_result/`. The JSON result files continue to represent the latest run.

The pipeline scans `data/images_test` (including subdirectories) and applies the entered question to each image. Qwen classifies the request as existence, counting, listing, attribute, OCR, or complex. Object detection/listing and supported object counting use YOLO; supported-target attribute and OCR tasks use YOLO to detect and crop the target, then send the crop to DeepSeek. If the target is missing, YOLO reports that; if a crop cannot be generated, the full image is sent to DeepSeek as a fallback. OCR without a specific supported target uses SmolVLM with the full image. Other full-image visual tasks use SmolVLM, while complex questions, future predictions, unsupported targets, and requests that do not fit another category go to DeepSeek. YOLO is skipped when none of the tasks needs detections or crops.

Each run saves machine-readable results (`results.json`, `summary.json`, and `errors.json`) and a uniquely named readable report in `outputs/images_test_result/`. The JSON files are overwritten with the latest run; previous text reports are kept.

Per-image reports show the model(s) used for each route: `DETECTIONS` uses YOLO, targeted `CROP` uses YOLO and Cloud (DeepSeek), `FULL_IMAGE` uses SmolVLM, and `CLOUD_ANSWER` uses Cloud (DeepSeek). Reports identify whether the cloud received a crop or the full image and include the saved crop path when applicable. If YOLO finds no target for an attribute or targeted OCR question, the pipeline reports that the requested information cannot be determined instead of asking a model to guess.

Payload reports only estimated edge-to-cloud request-body bytes. Local Qwen and YOLO processing contributes 0 bytes; DeepSeek routes include the serialized JSON request with its prompt and base64-encoded full image or target crop. This estimate excludes HTTP headers, TLS overhead, and cloud-to-edge response traffic, so it is not a wire-level network measurement.

The Qwen classifier requires `llama-cpp-python` and the `Qwen3-0.6B-Q4_0.gguf` weights at `models/Qwen/Qwen3-0.6B-Q4_0.gguf`; model weights are not stored in Git. To use the DeepSeek route, set `DEEPSEEK_API_KEY` in `models/DEEPSEEK/.env`. Optionally set `DEEPSEEK_MODEL` there (default: `deepseek-flash`) and `DEEPSEEK_MAX_TOKENS` (default: `4096`). Keep API keys private and do not commit the `.env` file.

Qwen routes requests based on semantic intent rather than fixed English keywords. There is no separate `cloud` task category: `complex` is the catch-all category and uses DeepSeek.
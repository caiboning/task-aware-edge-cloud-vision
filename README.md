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

The pipeline scans `data/images_test` (including subdirectories) and applies the entered question to each image. Qwen classifies the request as existence, counting, listing, attribute, OCR, or complex. Object detection/listing and supported object counting use YOLO; attributes use YOLO crops with SmolVLM, while OCR uses SmolVLM with the full image. Complex questions, future predictions, unsupported targets, and requests that do not fit another category go to DeepSeek. YOLO is skipped when none of the tasks needs detections or crops.

The Qwen classifier requires `llama-cpp-python` and the `Qwen3-0.6B-Q4_0.gguf` weights at `models/Qwen/Qwen3-0.6B-Q4_0.gguf`; model weights are not stored in Git. To use the DeepSeek route, set `DEEPSEEK_API_KEY` in `models/DEEPSEEK/.env`. Optionally set `DEEPSEEK_MODEL` there (default: `deepseek-flash`) and `DEEPSEEK_MAX_TOKENS` (default: `4096`). Keep API keys private and do not commit the `.env` file.

Qwen routes requests based on semantic intent rather than fixed English keywords. There is no separate `cloud` task category: `complex` is the catch-all category and uses DeepSeek.
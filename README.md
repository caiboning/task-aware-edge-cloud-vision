# Task-Aware Edge-to-Cloud Visual Communication for Physical AI

## Project Overview
This project explores task-aware semantic communication for Physical AI systems. Instead of always transmitting full-resolution images to the cloud, the system selects what visual information to process or transmit based on the user's query and available resources.

The system supports multiple strategies, including local inference using lightweight vision-language models (VLMs), transmitting object detection results, sending cropped image regions, or forwarding full images to a backend VLM.

Our goal is to reduce communication overhead and end-to-end latency while maintaining task accuracy. We evaluate different strategies based on task accuracy, transmitted bytes, and end-to-end latency.

## Installation
Python 3.11 is recommended. Create and activate a virtual environment:

```bash
python -m venv .env
```

On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .env/bin/activate
```

SmolVLM requires Ollama to run locally. If Ollama is not installed, install it using Windows PowerShell:

```bash
winget install Ollama.Ollama
```

After installation, restart the VS Code terminal and verify the installation:

```bash
ollama --version
```

Otherwise, install [Ollama](https://ollama.com/download) separately and pull the SmolVLM model:

```bash
ollama pull ahmadwaqar/smolvlm2-256m-video:q8_0
```

Install the Python dependencies and other models listed in `requirements.txt`and `download_models.py`:

```bash
python -m pip install -r requirements.txt
python download_models.py
```

Ensure the Ollama service is running. YOLO NCNN and Qwen GGUF model weights must be downloaded separately and placed at the paths specified in the project configuration.

## How to Use
Run both the task-aware pipeline and the full-image DeepSeek baseline on the same images and question:

```bash
python main.py
```

The program prompts for the question when it starts. To change the image or output directory, edit `IMAGE_DIR` or `OUTPUT_DIR` near the top of `main.py`. Both strategies use the same question and image directory.

The runs execute sequentially, not concurrently. For targeted OCR and attribute questions, the paired baseline uses the target identified by the task-aware pipeline to ask DeepSeek whether that object is visible first. It skips the follow-up when the object is absent or presence is uncertain; if present, it asks DeepSeek to transcribe text from that object or answer the requested attribute about that object only. Other baseline questions are sent directly, using one cloud request per image. A positive targeted OCR or attribute case uses two requests.

Run the paired task-aware and conditional baseline workflow through `python main.py`. To run only the full-image baseline independently, use `python baseline.py`; without a task-aware target plan, this standalone mode sends the supplied question directly.

The task-aware pipeline scans `IMAGE_DIR` (including subdirectories) and applies `QUESTION` from `main.py` to each image. Qwen classifies the request as existence, counting, listing, attribute, OCR, or complex. Object detection/listing and supported object counting use YOLO; supported-target attribute and OCR tasks use YOLO to detect and crop the target, then send the crop to DeepSeek. If the target is missing, YOLO reports that; if a crop cannot be generated, the full image is sent to DeepSeek as a fallback. OCR without a specific supported target uses SmolVLM with the full image. Other full-image visual tasks use SmolVLM, while complex questions, future predictions, unsupported targets, and requests that do not fit another category go to DeepSeek. YOLO is skipped when none of the tasks needs detections or crops.

Raw machine-readable results (`results.json` and `baseline_results.json`) are overwritten with the latest run. After both strategies finish, a timestamped `evaluation_report_YYYYMMDD_HHMMSS.txt` is generated with per-image answers, agreement/similarity, and an overall summary; previous evaluation reports are kept. Overall exact agreement combines binary matches with listing/OCR results that reach 100% similarity; lower similarity scores count as mismatches. Per-image rows retain Jaccard similarity for listing and normalized character-edit similarity for OCR. These metrics compare against DeepSeek, not human ground truth. The task-aware and baseline reports are also saved as timestamped text files. Summary, accuracy, evaluation, and error JSON files are not generated.

Latency reports separate image loading, Qwen model loading and inference, routing, YOLO model loading and inference, crop generation, SmolVLM preparation and calls, DeepSeek preparation and API calls, and other pipeline overhead. Summary component averages are per successful image; stages not used for an image count as 0. Model-load time is charged to the image that first triggers loading.

Per-image reports show the model(s) used for each route: `DETECTIONS` uses YOLO, targeted `CROP` uses YOLO and Cloud (DeepSeek), `FULL_IMAGE` uses SmolVLM, and `CLOUD_ANSWER` uses Cloud (DeepSeek). Reports identify whether the cloud received a crop or the full image and include the saved crop path when applicable. If YOLO finds no target for an attribute or targeted OCR question, the pipeline reports that the requested information cannot be determined instead of asking a model to guess.

Payload reports only estimated edge-to-cloud request-body bytes. Local Qwen and YOLO processing contributes 0 bytes; DeepSeek routes include the serialized JSON request with its prompt and base64-encoded full image or target crop. This estimate excludes HTTP headers, TLS overhead, and cloud-to-edge response traffic, so it is not a wire-level network measurement.

The Qwen classifier requires `llama-cpp-python` and the `Qwen3-0.6B-Q4_0.gguf` weights at `models/Qwen/Qwen3-0.6B-Q4_0.gguf`; model weights are not stored in Git. To use the DeepSeek route, set `DEEPSEEK_API_KEY` in `models/DEEPSEEK/.env`. Optionally set `DEEPSEEK_MODEL` there (default: `deepseek-flash`) and `DEEPSEEK_MAX_TOKENS` (default: `4096`). Keep API keys private and do not commit the `.env` file.

Qwen routes requests based on semantic intent rather than fixed English keywords. There is no separate `cloud` task category: `complex` is the catch-all category and uses DeepSeek.
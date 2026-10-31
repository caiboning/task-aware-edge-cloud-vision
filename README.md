# Task-Aware Edge-to-Cloud Visual Communication for Physical AI

## Project Overview
This project explores task-aware semantic communication for Physical AI systems. Instead of always transmitting full-resolution images to the cloud, the system selects what visual information to process or transmit based on the user's query and available resources.

The system supports multiple strategies, including local inference using lightweight vision-language models (VLMs), transmitting object detection results, sending cropped image regions, or forwarding full images to a backend VLM.

Our goal is to reduce communication overhead and end-to-end latency while maintaining task accuracy. We evaluate different strategies based on task accuracy, transmitted bytes, and end-to-end latency.

## How to Use
The planned application entry point is:

```bash
python pipeline.py
```

Once the inference pipeline is implemented, users will be able to:

1. Upload an image and enter a visual question.
2. Select a processing strategy (local VLM, object detection results, cropped regions, or full-image transmission).
3. View the model's response.
4. Compare task accuracy, communication cost, and latency across strategies.

**Status:** Under development. Model integration and edge-cloud communication are not yet fully implemented.
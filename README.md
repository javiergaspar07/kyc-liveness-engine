# 🛡️ KYC Liveness Engine

A real-time, asynchronous streaming pipeline designed for Know Your Customer (KYC) identity verification. This engine captures webcam feeds via WebSockets, isolates faces using lightweight CPU detection, and runs a deep learning model to detect spoofing attempts (liveness verification) with low latency.

## 🚀 Tech Stack

* **Backend Framework:** FastAPI, Uvicorn (Asynchronous Python)
* **Communication:** WebSockets
* **Face Detection (Gatekeeper):** Google MediaPipe (Tasks API)
* **Liveness Classification:** PyTorch, `timm` (EfficientNet-B0)
* **Frontend Test Client:** HTML5, JavaScript (Canvas API, MediaDevices)

## ⚙️ Core Architecture & Optimizations

* **Traffic Control (Frame Dropping):** To prevent server overload and memory collapse, the WebSocket processes only 1 out of every 5 incoming frames (approx. 6 FPS). This maintains real-time responsiveness without bottlenecking the GPU.
* **Two-Phase Pipeline:** 1. **CPU Gatekeeper:** MediaPipe quickly scans the frame. If no face (or multiple faces) are detected, the frame is instantly discarded before reaching the heavy AI model.
  2. **GPU Inference:** PyTorch processes the strictly cropped and normalized face tensor. The inference is wrapped in `torch.no_grad()` to ensure zero mathematical history is tracked, preventing memory leaks during continuous streaming.
* **Strict Confidence Threshold:** Enforces a `>= 0.85` Softmax probability threshold to pass the liveness check, prioritizing systemic security against False Positives (spoofs).
* **Safe Matrix Slicing:** Dynamically bounds bounding box coordinates using `min()` and `max()` functions to guarantee matrix operations never attempt to read out-of-bounds pixels.

## 🛠️ Installation & Setup

**1. Clone the repository and set up your environment**
```bash
git clone [https://github.com/yourusername/kyc-liveness-engine.git](https://github.com/yourusername/kyc-liveness-engine.git)
cd kyc-liveness-engine

# It is recommended to use a virtual environment (e.g., Conda or venv)
# conda create -n ai python=3.10
# conda activate ai
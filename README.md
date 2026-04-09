# 🛡️ KYC Liveness Engine

A real-time, asynchronous streaming pipeline designed for Know Your Customer (KYC) identity verification. This engine captures webcam feeds via WebSockets, isolates faces using lightweight CPU detection, and runs a deep learning model to detect spoofing attempts (liveness verification) with low latency.

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python), WebSockets, Uvicorn
* **Computer Vision:** OpenCV, MediaPipe (Face Mesh for Liveness)
* **Deep Learning:** PyTorch, FaceNet (Inception Resnet V1 for 512D Embeddings)
* **Database:** PostgreSQL + `pgvector` extension, SQLAlchemy (Async)
* **Frontend Clients:** Vanilla HTML5, CSS3, JavaScript (Canvas API for frame extraction)

## ⚙️ Core Architecture & Optimizations

* **Zero-Trust Enrollment & Authentication:** Abandons static image payloads (which are vulnerable to injection and replay attacks) in favor of secure WebSockets. Facial extraction *only* occurs after the user proves they are a live human in real-time.
* **Dynamic 3D Liveness State Machine:** Requires users to pass randomized, multi-step head-pose challenges (Pitch/Yaw/Roll) before granting access.
* **Mathematically Pure Pose Estimation:** Implements custom coordinate system alignment to correct OpenCV/MediaPipe *Gimbal Lock* mismatch, ensuring highly accurate head pose tracking without relying on arithmetic hacks.
* **"Sweet Spot" Biometric Capture:** Prevents degraded facial extractions by forcing the user to return to a perfect `(0,0,0)` center pose before capturing the frame for the Deep Learning model.
* **Vector Search Database:** Utilizes PostgreSQL with `pgvector` to store 512-dimensional facial embeddings safely.
* **Atomic Transactions:** Protects the database against race conditions and "Database Poisoning" during enrollment.
## 🛠️ Installation & Setup

**1. Clone the repository and set up your environment**
```bash
git clone [https://github.com/javiergaspar07/kyc-liveness-engine.git](https://github.com/javiergaspar07/kyc-liveness-engine.git)
cd kyc-liveness-engine

# It is recommended to use a virtual environment (e.g., Conda or venv)
# conda create -n ai python=3.10
# conda activate ai
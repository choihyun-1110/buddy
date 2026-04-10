# Pill Buddy

Real-time pill ingestion detection system using computer vision.  
Detects whether a person has taken a pill by analyzing hand, mouth, and cup interactions — without directly detecting the pill itself.

---

## How It Works

Pills are hidden in the hand and invisible to the camera.  
Instead of detecting the pill directly, the system tracks a behavioral sequence:

```
IDLE → HAND_TO_MOUTH → CUP_TO_MOUTH → VERIFY → INGESTION_COMPLETE ✅
```

| State | Condition |
|-------|-----------|
| `HAND_TO_MOUTH` | Hand overlaps mouth ROI for ≥ 3 frames |
| `CUP_TO_MOUTH` | Cup detected near mouth for ≥ 2 frames |
| `VERIFY` | Cup leaves mouth → open mouth wide to confirm |
| `INGESTION_COMPLETE` | Mouth open verified → `O` (done) |

---

## Components

| Module | Role |
|--------|------|
| `mouth_roi_tracker/` | FaceMesh mouth region tracking (EMA smoothed, with fallback) |
| `pill_ingestion/hand_tracker.py` | MediaPipe 21-keypoint hand detection + grip detection |
| `pill_ingestion/cup_detector.py` | YOLOv8n cup/bottle detection |
| `pill_ingestion/swallow_estimator.py` | Lip closure (swallow) + mouth open (verify) detection |
| `pill_ingestion/ingestion_fsm.py` | FSM state machine |
| `pill_recognition_ai/` | YOLOv8n pill classifier (22 classes, Raspberry Pi optimized) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Webcam (real-time)
python run_pill_ingestion.py

# Video file
python run_pill_ingestion.py --video dataset/IMG_2273.mov

# Save output video
python run_pill_ingestion.py --video dataset/IMG_2273.mov --save output.mp4

# Hand-only mode (no cup required)
python run_pill_ingestion.py --no-cup
```

**Controls**
- `q` — quit
- `space` — pause / resume
- `r` — reset FSM

---

## Models

Download and place in `models/`:

| File | Source |
|------|--------|
| `face_landmarker.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) |
| `hand_landmarker.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| `yolov8n.pt` | `pip install ultralytics` → auto-downloaded on first run |

See [`models/README.md`](models/README.md) for details.

---

## Target Hardware

- Development: macOS (Apple Silicon / Intel)
- Deployment: **Raspberry Pi 4** (4GB RAM)
  - ONNX Runtime CPU inference
  - Input 416×416, batch size 1
  - Target: ≥ 5 FPS, ≤ 200ms latency

---

## Requirements

```
opencv-python>=4.8.0
mediapipe==0.10.30
numpy>=1.24.0
ultralytics>=8.0.0
onnxruntime>=1.15.0
```

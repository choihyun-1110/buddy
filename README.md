# Pill Buddy

Real-time pill ingestion detection system using computer vision.  
Detects whether a person has taken a pill by analyzing hand, mouth, and cup interactions — **without directly detecting the pill itself**.

> Demo: place the GIF here after recording with `--save output.mp4` and converting to GIF.

---

## How It Works

Pills are hidden in the hand and invisible to the camera.  
Instead of detecting the pill directly, the system tracks a **behavioral sequence**:

```
IDLE → HAND_TO_MOUTH → CUP_TO_MOUTH → VERIFY → INGESTION_COMPLETE ✅
```

| State | Trigger |
|-------|---------|
| `HAND_TO_MOUTH` | Hand overlaps mouth ROI for ≥ 3 frames |
| `CUP_TO_MOUTH` | Cup detected near mouth for ≥ 2 frames |
| `VERIFY` | Cup leaves mouth → user opens mouth wide to confirm |
| `INGESTION_COMPLETE` | Mouth-open verified → decision `O` (done) |

Suspicious patterns (repeated FAIL, face hidden, skipped VERIFY) trigger a **Slack alert** to caregivers.

---

## Components

| Module | Role |
|--------|------|
| `mouth_roi_tracker/` | FaceMesh mouth-region tracking (EMA smoothed, with fallback) |
| `pill_ingestion/hand_tracker.py` | MediaPipe 21-keypoint hand detection + grip detection |
| `pill_ingestion/cup_detector.py` | YOLOv8n cup/bottle detection (ONNX Runtime, imgsz=320) |
| `pill_ingestion/swallow_estimator.py` | Lip-closure (swallow) + mouth-open (verify) detection |
| `pill_ingestion/ingestion_fsm.py` | FSM state machine |
| `pill_ingestion/alert_manager.py` | Suspicion scoring + Slack webhook alert |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

Place the following files in `models/`:

| File | Source |
|------|--------|
| `face_landmarker.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) |
| `hand_landmarker.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| `yolov8n.pt` | auto-downloaded on first run via `ultralytics` |

Then export YOLOv8 to ONNX for faster inference:

```bash
python scripts/export_yolo_onnx.py
```

### 3. Run

```bash
# Webcam (real-time)
python run_pill_ingestion.py

# Your own video
python run_pill_ingestion.py --video /path/to/video.mov

# Save output video
python run_pill_ingestion.py --video /path/to/video.mov --save output.mp4

# Hand-only mode (no cup required)
python run_pill_ingestion.py --no-cup

# Benchmark mode (FPS + Pi4 estimate)
python run_pill_ingestion.py --video /path/to/video.mov --benchmark
```

> **Note:** Sample video is not included in this repo due to privacy. Use your own recording or a webcam.

**Controls:** `q` quit · `space` pause/resume · `r` reset FSM

---

## Slack Alert (optional)

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
python run_pill_ingestion.py --patient-id "room-101"
```

Alerts fire when suspicion score ≥ 5 (repeated failures, skipped verification, face hidden).

---

## Performance

Measured on Apple M5 Mac, estimated for Raspberry Pi 4 (×9 inference scaling):

| Component | Mac | Pi4 est. |
|-----------|-----|----------|
| FaceMesh | ~7ms | ~63ms |
| Hand | ~10ms | ~90ms |
| YOLO cup | ~2ms | ~18ms |
| **Total inference** | **~19ms** | **~171ms** |
| **Est. FPS** | **~16** | **~3.9** |

YOLO runs every 3 frames and Hand every 2 frames to reduce Pi load.  
Run `--benchmark` flag to measure on your own hardware.

---

## Target Hardware

- Development: macOS (Apple Silicon / Intel)
- Deployment: **Raspberry Pi 4** (4GB RAM)
  - ONNX Runtime CPU-only inference
  - YOLOv8n imgsz=320, batch=1
  - Goal: ≥ 5 FPS, ≤ 200ms latency

---

## Tests

```bash
pytest tests/
```

Covers FSM state transitions, mouth ROI tracking, swallow estimator, and alert manager.

---

## Requirements

```
opencv-python>=4.8.0
mediapipe==0.10.30
numpy>=1.24.0
ultralytics>=8.0.0
onnxruntime>=1.15.0
pytest>=7.0.0
```

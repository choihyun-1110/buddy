# 모델 파일 (로컬 전용, API 호출 없음)

실행 중 네트워크/API 호출은 하지 않습니다. 아래 파일들은 **프로젝트 루트의 `models/`** 에 두세요 (이 README와 같은 폴더).

## face_landmarker.task (필수 – Mouth ROI)

```bash
cd models
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

## hand_landmarker.task (선택 – 복용 시퀀스 Hand 검출)

`run_pill_ingestion.py` 에서 손 검출을 쓰려면 한 번 받아 두세요. 없으면 손 검출은 비활성(빈 결과)입니다.

```bash
cd models
curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

## yolov8n.pt (선택 – 컵/병 검출)

컵 검출은 **YOLO(v8 nano)** 를 사용합니다. **이 파일을 `models/` 에 두면** 실행 시 네트워크 없이 로컬에서만 추론합니다.  
없으면 ultralytics가 **최초 1회만** 인터넷에서 받아 캐시(~/.config/Ultralytics 등)에 저장합니다.

- **프로젝트 안에 두고 완전 오프라인으로 쓰려면** (권장):

```bash
cd models
curl -L -o yolov8n.pt "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt"
```

또는 [Hugging Face Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) 에서 `yolov8n.pt` 를 받아 `models/yolov8n.pt` 에 넣으면 됩니다.

- **컵이 안 잡힐 때**: COCO 클래스 39=bottle, 40=wine_glass, 41=cup 만 검출합니다. 각도/크기/조명에 따라 confidence 가 낮을 수 있어, `pill_ingestion/cup_detector.py` 의 `conf_threshold`(기본 0.18)를 더 낮춰 보세요. 컵 없이 손만으로 복용 인정하려면 `run_pill_ingestion.py --no-cup` 를 쓰면 됩니다.

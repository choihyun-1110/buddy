# 설치 안내

## 1. 가상환경 사용 (권장)

```bash
cd /Users/hyun/dev/buddy
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Face Landmarker 모델 (로컬 파일만 사용, API/자동 다운로드 없음)

Mouth ROI Tracker는 MediaPipe **Tasks API (FaceLandmarker)** 를 사용하며, **로컬 모델 파일만** 사용합니다 (실행 중 네트워크/API 호출 없음).

모델 파일을 한 번만 받아서 프로젝트에 두세요:

```bash
mkdir -p models
curl -L -o models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

또는 브라우저에서 위 URL 로 받은 파일을 `models/face_landmarker.task` 에 저장하면 됩니다.

## 3. 컵 검출용 YOLO 모델 (선택, 로컬 추론만)

컵/병 검출은 **ultralytics YOLO**를 쓰며, **API 호출 없이** 로컬에서만 추론합니다.

- **`models/yolov8n.pt` 를 두면** 이 파일만 사용합니다 (네트워크 불필요).
- 없으면 최초 실행 시 ultralytics가 한 번만 인터넷에서 받아 캐시에 저장합니다.

완전 오프라인으로 쓰려면 `models/README.md` 의 yolov8n.pt 안내대로 받아 두세요.

## 4. mediapipe 설치 오류가 날 때 (SyntaxError: invalid character '∂')

일부 환경에서 mediapipe 최신 버전 설치 시 패키지 안의 테스트 파일 때문에 에러가 납니다. 아래를 순서대로 시도하세요.

### 1) 문제 버전 제거 후 지정 버전 설치

```bash
pip3 uninstall mediapipe -y
pip3 install mediapipe==0.10.30
pip3 install opencv-python numpy pytest
```

### 2) 그래도 실패하면: 가상환경 사용 (권장)

```bash
cd /Users/hyun/dev/buddy
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

이후 실행할 때마다 `source .venv/bin/activate` 한 다음 `python run_mouth_roi_example.py --video` 하면 됩니다.

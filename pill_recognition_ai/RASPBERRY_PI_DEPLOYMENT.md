# 라즈베리파이 배포 가이드

## 📋 목차
1. [개요](#개요)
2. [PC에서 ONNX 변환](#pc에서-onnx-변환)
3. [PC에서 시뮬레이션 테스트](#pc에서-시뮬레이션-테스트)
4. [라즈베리파이 환경 설정](#라즈베리파이-환경-설정)
5. [라즈베리파이에서 테스트](#라즈베리파이에서-테스트)
6. [성능 최적화 팁](#성능-최적화-팁)

## 개요

라즈베리파이는 GPU가 없으므로 CPU만으로 추론을 실행해야 합니다. ONNX 형식으로 변환하여 ONNX Runtime으로 실행합니다.

**예상 성능:**
- 라즈베리파이 4 (4GB): 약 200-500ms/이미지 (2-5 FPS)
- 라즈베리파이 5 (4GB): 약 100-300ms/이미지 (3-10 FPS)

## PC에서 ONNX 변환

### 1. 모델을 ONNX로 변환

```bash
cd C:\Users\User\Desktop\buudy\pill_recognition_ai
pill_ai_env\Scripts\activate
python -c "from ultralytics import YOLO; model = YOLO('results/training_20251031_201041/best_model.pt'); model.export(format='onnx', imgsz=640, dynamic=False, simplify=True, opset=12)"
```

변환된 모델은 `runs/detect/weights/best.onnx`에 저장됩니다.

### 2. PC에서 시뮬레이션 테스트

라즈베리파이가 없어도 PC에서 CPU 모드로 테스트할 수 있습니다:

```bash
# ONNX 변환 및 테스트
python raspberry_pi_test.py --model results/training_20251031_201041/best_model.pt --export --images ../real_image0.jpeg ../real_image1.jpeg

# 이미 ONNX 파일이 있으면
python raspberry_pi_test.py --model runs/detect/weights/best.onnx --images ../real_image0.jpeg ../real_image1.jpeg
```

이 명령은:
- 모델을 ONNX로 변환
- CPU 모드로 추론 실행
- 추론 시간, CPU 사용률, 메모리 사용량 측정
- 라즈베리파이 호환성 평가

## 라즈베리파이 환경 설정

### 1. 시스템 업데이트

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Python 가상환경 생성

```bash
python3 -m venv pill_env
source pill_env/bin/activate
pip install --upgrade pip
```

### 3. 필수 패키지 설치

```bash
# 시스템 패키지
sudo apt install python3-dev libopenblas-dev libomp-dev

# Python 패키지
pip install onnxruntime numpy opencv-python pillow
```

**참고:** 라즈베리파이는 GPU가 없으므로 `onnxruntime` (CPU 버전)만 설치하면 됩니다.

### 4. 모델 및 스크립트 전송

PC에서 라즈베리파이로 파일 전송:

```bash
# SCP 사용 (Windows PowerShell)
scp C:/Users/User/Desktop/buudy/pill_recognition_ai/runs/detect/weights/best.onnx pi@라즈베리IP:/home/pi/pill_model/
scp C:/Users/User/Desktop/buudy/pill_recognition_ai/raspberry_pi_inference.py pi@라즈베리IP:/home/pi/pill_model/
```

또는 USB 드라이브나 네트워크 공유 폴더 사용 가능.

## 라즈베리파이에서 테스트

### 1. 기본 추론 테스트

```bash
cd ~/pill_model
source pill_env/bin/activate
python raspberry_pi_inference.py --model best.onnx --image test_image.jpg --output result.jpg
```

### 2. 성능 측정

```bash
# 여러 이미지로 성능 테스트
for img in test_images/*.jpg; do
    python raspberry_pi_inference.py --model best.onnx --image "$img" --output "results/$(basename $img)" --json "results/$(basename $img .jpg).json"
done
```

### 3. 실시간 카메라 추론 (선택)

```python
# camera_inference.py
import cv2
import onnxruntime as ort
import numpy as np

# 모델 로드
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 전처리 및 추론
    # ... (raspberry_pi_inference.py 참고)
    
    # 결과 표시
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 성능 최적화 팁

### 1. 이미지 크기 줄이기

640x640 대신 416x416이나 320x320 사용:

```python
# ONNX 변환 시
model.export(format='onnx', imgsz=416, ...)

# 추론 시
img_batch = preprocess_image(image_path, (416, 416))
```

### 2. 모델 양자화 (고급)

INT8 양자화로 모델 크기와 속도 개선:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "best.onnx",
    "best_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### 3. ONNX Runtime 최적화

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_cpu_mem_arena = True
sess_options.enable_mem_pattern = True
```

### 4. 입력 전처리 최적화

OpenCV 대신 PIL 사용 (더 빠를 수 있음):

```python
from PIL import Image
import numpy as np

img = Image.open(image_path).resize((640, 640))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = img_array.transpose(2, 0, 1)[None, :]
```

## 문제 해결

### 1. 메모리 부족

- 이미지 크기 줄이기
- 배치 크기 1로 유지
- 다른 프로세스 종료

### 2. 추론 속도가 너무 느림

- 이미지 크기 줄이기 (640 -> 416)
- 모델 양자화 시도
- 라즈베리파이 5 사용 고려

### 3. ONNX Runtime 오류

- ONNX Runtime 버전 확인: `pip install onnxruntime==1.17.1`
- ONNX opset 버전 확인 (opset=12 권장)

## 참고사항

- **모델 파일 크기:** 약 8-10MB (ONNX)
- **메모리 사용량:** 약 200-500MB
- **CPU 사용률:** 100% (단일 스레드)
- **추론 시간:** 라즈베리파이 4 기준 200-500ms/이미지

실시간 처리가 필요한 경우, 이미지 크기를 줄이거나 라즈베리파이 5를 사용하는 것을 권장합니다.


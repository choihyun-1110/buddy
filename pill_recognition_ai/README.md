# 알약 인식 AI 모델 개발 프로젝트

RTX 3060Ti 8GB VRAM 환경에 최적화된 YOLOv8 기반 알약 인식 AI 모델 개발 프로젝트입니다.

## 🎯 프로젝트 목표

- **입력**: 알약이 포함된 RGB 이미지 (PNG 형식)
- **출력**: 각 알약의 Bounding Box 좌표 + 클래스 이름
- **목표 환경**: RTX 3060Ti 8GB VRAM에서 실시간 추론 가능

## 🏗️ 프로젝트 구조

```
pill_recognition_ai/
├── configs/                 # 설정 파일
│   ├── data_config.yaml    # 데이터 설정
│   └── training_config.yaml # 학습 설정
├── dataset/                 # 데이터셋 (자동 생성)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                  # 모델 관련 코드
│   ├── yolo_trainer.py     # YOLOv8 학습
│   └── model_evaluator.py  # 모델 평가
├── utils/                   # 유틸리티
│   ├── data_preprocessing.py # 데이터 전처리
│   ├── data_augmentation.py  # 데이터 증강
│   └── gpu_optimizer.py     # GPU 최적화
├── results/                 # 학습 결과
├── logs/                    # 로그 파일
├── pill_ai_env/            # 가상환경
├── requirements.txt        # 의존성 패키지
└── main.py                 # 메인 실행 스크립트
```

## 🚀 빠른 시작

### 1. 가상환경 활성화

```bash
# Windows
pill_ai_env\Scripts\activate

# Linux/Mac
source pill_ai_env/bin/activate
```

### 2. 빠른 테스트 실행

```bash
python main.py --mode test
```

이 명령어는 데이터 전처리만 실행하여 프로젝트가 정상 작동하는지 확인합니다.

### 3. 전체 파이프라인 실행

```bash
python main.py --mode full
```

## 📋 실행 모드

### 테스트 모드 (권장)
```bash
python main.py --mode test
```
- 데이터 전처리만 실행
- 프로젝트 설정 확인용

### 전체 파이프라인
```bash
python main.py --mode full
```
- 데이터 전처리 → 증강 → 학습 → 평가 → 최적화
- 전체 과정 자동 실행

### 개별 단계 실행
```bash
# 데이터 전처리만
python main.py --mode preprocess

# 학습만 (전처리 완료 후)
python main.py --mode train

# 평가만 (학습 완료 후)
python main.py --mode evaluate
```

### 특정 단계 건너뛰기
```bash
# 2단계(증강)와 5단계(최적화) 건너뛰기
python main.py --mode full --skip 2 5
```

## ⚙️ RTX 3060Ti 최적화 설정

### GPU 메모리 최적화
- **배치 크기**: 8 (8GB VRAM에 최적화)
- **이미지 크기**: 640x640
- **모델**: YOLOv8n (가장 가벼운 모델)
- **옵티마이저**: AdamW (메모리 효율적)

### 권장 하드웨어 사양
- **GPU**: RTX 3060Ti 8GB VRAM
- **RAM**: 16GB 이상
- **저장공간**: 10GB 이상 여유 공간

## 📊 데이터셋 정보

### 원본 데이터
- **총 이미지**: 920개
- **클래스 수**: 22개 (알약 종류별)
- **이미지 형식**: PNG
- **특징**: 대부분 단일 알약 포함

### 처리된 데이터
- **분할 비율**: Train(80%) / Val(10%) / Test(10%)
- **형식**: YOLO 형식 (정규화된 바운딩 박스)
- **증강**: 다중 알약 합성 이미지 생성

## 🎯 모델 성능 목표

### 학습 목표
- **mAP@0.5**: 0.8 이상
- **Precision**: 0.85 이상
- **Recall**: 0.8 이상

### 추론 성능 목표
- **FPS**: 15 FPS 이상
- **지연시간**: 100ms 이하
- **메모리 사용량**: 6GB 이하

## 🔧 주요 기능

### 1. 데이터 전처리
- PNG 이미지를 YOLO 형식으로 변환
- 자동 바운딩 박스 추출
- 데이터셋 분할 (train/val/test)

### 2. 데이터 증강
- 여러 알약을 하나의 이미지에 합성
- 다양한 배경 텍스처 생성
- 겹침 방지 알고리즘

### 3. 모델 학습
- YOLOv8 기반 학습
- RTX 3060Ti 최적화 설정
- 실시간 메모리 모니터링

### 4. 모델 평가
- mAP, Precision, Recall 측정
- 클래스별 성능 분석
- 추론 속도 측정

### 5. 모델 최적화
- ONNX 변환
- GPU 메모리 최적화
- 실시간 추론 최적화

## 📈 모니터링 및 로깅

### Weights & Biases
- 실시간 학습 모니터링
- 하이퍼파라미터 추적
- 모델 성능 비교

### TensorBoard
- 학습 곡선 시각화
- 손실 함수 추적
- 메트릭 모니터링

### GPU 모니터링
- 실시간 VRAM 사용량
- GPU 온도 및 사용률
- 메모리 최적화 제안

## 🐛 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
# configs/training_config.yaml에서 batch_size를 4로 변경
```

### 학습 속도가 느린 경우
```bash
# 워커 수 조정
# configs/training_config.yaml에서 workers를 2로 변경
```

### 데이터 로딩 오류
```bash
# 데이터 경로 확인
# configs/data_config.yaml에서 source_path 확인
```

## 📝 사용 예시

### 기본 사용법
```python
from main import PillRecognitionPipeline

# 파이프라인 초기화
pipeline = PillRecognitionPipeline()

# 전체 파이프라인 실행
pipeline.run_full_pipeline()

# 특정 단계만 실행
pipeline.step1_data_preprocessing()
```

### 커스텀 설정
```python
# 설정 파일 수정 후 실행
pipeline = PillRecognitionPipeline("custom_config.yaml")
pipeline.run_full_pipeline()
```

## 🤝 기여하기

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서화 개선

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하면 이슈를 생성하거나 다음을 확인하세요:
- GPU 드라이버 업데이트
- CUDA 버전 호환성
- 메모리 사용량 모니터링

# 알약 인식 AI 프로젝트 보고서

## 📋 프로젝트 개요

**프로젝트명**: 알약 인식 AI 파이프라인  
**목표**: YOLOv8 기반 다중 알약 인식 모델 개발 및 라즈베리파이 배포  
**기간**: 2024년 10월 ~ 11월

---

## 🎯 주요 작업 내용

### 1. Git 저장소 설정 및 관리

- **저장소 초기화**: `git init`
- **원격 저장소 연결**: `https://github.com/choihyun-1110/buddy.git`
- **`.gitignore` 설정**: 대용량 데이터셋, 모델 파일, 가상환경 제외
- **초기 커밋 및 푸시**: 프로젝트 구조 정리

### 2. 데이터 증강 개선

#### 🔧 발견된 문제
- **버그**: `utils/data_augmentation.py`에서 다중 알약 합성 시 클래스 ID가 잘못 매핑됨
  - 원인: `original_class_id % 2`로 클래스를 0 또는 1로 제한
  - 결과: 22개 클래스 중 일부가 잘못된 레이블로 학습됨

#### ✅ 해결 방법
- 클래스 ID 매핑 로직 제거
- 원본 클래스 ID를 그대로 유지하도록 수정
- 데이터 재증강: `dataset_augmented_v3` 생성

### 3. 모델 학습 및 평가

#### 학습 설정
- **모델**: YOLOv8n (YOLOv11n)
- **데이터셋**: `dataset_augmented_v3`
  - 클래스 수: 22개
  - Train/Val/Test 분할: 표준 YOLO 형식
- **최적 모델**: `results/training_20251031_201041/best_model.pt`

#### 평가 결과
- **Test Set 성능**: mAP, Precision, Recall 측정
- **시각화**: 다중 알약 이미지에 대한 추론 결과 확인
- **문제 해결**: 다중 알약 감지 성능 개선 확인

### 4. 라즈베리파이 배포 준비

#### ONNX 변환
- **변환된 모델**: `best_model.onnx` (11.7 MB)
- **설정**: 
  - 이미지 크기: 640x640
  - Opset: 12
  - 정적 크기 (dynamic=False)
  - 모델 단순화 (simplify=True)

#### 성능 테스트
- **PC CPU 모드 시뮬레이션**:
  - 추론 시간: ~35-40ms
  - 예상 FPS: 25-30
  - CPU 사용률: ~20%
- **라즈베리파이 예상 성능**:
  - 라즈베리파이 4: 200-500ms/이미지 (2-5 FPS)
  - 라즈베리파이 5: 100-300ms/이미지 (3-10 FPS)

#### 생성된 파일
- `raspberry_pi_test.py`: PC에서 라즈베리파이 환경 시뮬레이션
- `raspberry_pi_inference.py`: 라즈베리파이에서 실행할 추론 스크립트
- `test_raspberry_pi_inference.py`: 특정 이미지 추론 및 시각화
- `RASPBERRY_PI_DEPLOYMENT.md`: 배포 가이드 문서

### 5. 추론 파이프라인 구현

#### 주요 기능
- **ONNX Runtime CPU 모드**: GPU 없이 추론 실행
- **YOLO 출력 후처리**: 
  - 정규화된 좌표 처리
  - NMS (Non-Maximum Suppression) 적용
  - 바운딩 박스 시각화
- **좌표 변환**: 픽셀 좌표 ↔ 정규화 좌표 자동 감지

#### 결과 시각화
- 바운딩 박스 그리기
- 클래스 이름 및 신뢰도 표시
- 다중 알약 동시 감지 확인

---

## 📁 프로젝트 구조

```
pill_recognition_ai/
├── main.py                          # 메인 파이프라인
├── configs/
│   ├── dataset_augmented_v3.yaml    # 데이터셋 설정
│   └── training_config.yaml         # 학습 설정
├── utils/
│   ├── data_augmentation.py         # 데이터 증강 (수정됨)
│   ├── gpu_optimizer.py            # GPU 최적화
│   └── ...
├── models/
│   └── yolo_trainer.py             # YOLO 학습 클래스
├── results/
│   └── training_20251031_201041/
│       └── best_model.pt           # 최종 모델
│       └── best_model.onnx         # ONNX 변환 모델
├── dataset_augmented_v3/            # 증강된 데이터셋
│   ├── train/
│   ├── val/
│   └── test/
└── raspberry_pi_*.py               # 라즈베리파이 배포 스크립트
```

---

## 🔑 주요 성과

### 기술적 성과
1. ✅ **다중 알약 감지 성능 개선**: 데이터 증강 버그 수정으로 정확도 향상
2. ✅ **모델 최적화**: ONNX 변환으로 배포 가능한 크기 (11.7 MB)
3. ✅ **크로스 플랫폼 지원**: PC에서 테스트, 라즈베리파이 배포 준비 완료

### 코드 품질
1. ✅ **재사용 가능한 스크립트**: 모듈화된 추론 및 시각화 코드
2. ✅ **문서화**: 배포 가이드 및 사용법 문서 작성
3. ✅ **버전 관리**: Git을 통한 코드 관리

---

## 🚀 다음 단계

### 단기 목표
- [ ] 라즈베리파이 실제 환경 테스트
- [ ] 추론 속도 최적화 (이미지 크기 조정, 양자화)
- [ ] 실시간 카메라 추론 구현

### 장기 목표
- [ ] 모델 성능 추가 개선
- [ ] 더 많은 알약 클래스 추가
- [ ] Edge AI 디바이스 최적화

---

## 📊 기술 스택

- **프레임워크**: Ultralytics YOLOv8/YOLOv11
- **언어**: Python 3.10
- **딥러닝**: PyTorch
- **배포**: ONNX Runtime
- **데이터 처리**: OpenCV, NumPy
- **버전 관리**: Git, GitHub

---

## 📝 참고 자료

- **데이터셋**: MFDS_AI_Development_Pill_Image_Dataset
- **모델**: YOLOv8n (nano 버전)
- **문서**: `RASPBERRY_PI_DEPLOYMENT.md` - 라즈베리파이 배포 가이드

---

## 👤 작성자

**Choi Hyun**  
**이메일**: chlgusla1@1yonsei.ac.kr  
**저장소**: https://github.com/choihyun-1110/buddy.git

---

**작성일**: 2024년 11월  
**마지막 업데이트**: 2024년 11월 1일


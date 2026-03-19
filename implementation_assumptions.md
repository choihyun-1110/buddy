# Mouth ROI Tracker – 명세에 없이 구현한 사항

`gpt가써준명세서.md`에는 없지만 구현 과정에서 정한 가정·선택 사항을 정리한 문서입니다.

---

## 1. 상수·파라미터 값

| 항목 | 명세 | 구현에서 사용한 값 |
|------|------|-------------------|
| **N** (얼굴 미검출 시 이전 ROI 유지 최대 프레임) | “최대 N 프레임”만 언급 | `max_frames_no_face = 5` |
| **K** (랜드마크 실패 시 이전 ROI 유지 최대 프레임) | “최대 K 프레임”만 언급 | `max_frames_keep_roi = 10` |
| **Landmark confidence threshold** | “confidence < threshold”만 언급 | `landmark_confidence_threshold = 0.5` (Face Landmarker 옵션 min_face_detection_confidence 등으로 전달) |
| **EMA alpha** | “0.3~0.6, 초기값 0.5” | `ema_alpha = 0.5` |

---

## 2. 라이브러리·환경

- **Face Landmarker**: MediaPipe Tasks API (`FaceLandmarker`). 구 API(`mp.solutions`)는 최신 패키지에서 제거되어 사용하지 않음. 모델 `models/face_landmarker.task` 는 **로컬 파일만** 사용(실행 중 API/네트워크 호출 없음). 없으면 자동 다운로드하지 않고 `FileNotFoundError` 로 수동 다운로드 안내(SETUP.md, models/README.md 참고).
- **이미지 처리**: OpenCV (`cv2`), BGR 포맷 (명세 “BGR image”와 동일)
- **언어**: Python 3.9+ (타입 힌트 `list[float]` 등 사용)
- **의존성**: `requirements.txt` (opencv-python, mediapipe, numpy, pytest, ultralytics)

---

## 3. 입 랜드마크 인덱스 (Outer Lip)

명세에는 “Outer lip landmarks”만 있고, MediaPipe 468 인덱스 중 어떤 것을 쓸지 없음.

- **구현**: `OUTER_LIP_LANDMARK_INDICES`에 Face Landmarker(468/478 랜드마크) 입술 외곽 인덱스 목록을 둠.
- **출처**: 기존 Face Mesh lip contour 인덱스를 참고한 집합. Tasks API Face Landmarker와 호환되도록 사용 중.

---

## 4. 얼굴이 없을 때 “이전 ROI”도 없는 경우

명세: “실패 시: 이전 프레임 ROI 유지 (최대 N 프레임)”만 기술.

- **구현**: 이전 ROI가 없거나 N 프레임을 넘긴 경우, **기본 ROI**를 사용함.  
  `[w*0.35, h*0.5, w*0.65, h*0.9]` (화면 중앙~하단) 형태의 fallback ROI를 반환.

---

## 5. State Machine 확장

명세: `roi_valid`, `roi_mode`, `frames_since_valid` 언급.

- **구현**: “Fail 1~K frame → 이전 ROI 유지, K 초과 → FALLBACK”를 구현하기 위해 **`frames_since_landmark_ok`**를 추가함.  
  - `frames_since_valid`: 얼굴이 **안** 나온 연속 프레임 수 (N과 비교).  
  - `frames_since_landmark_ok`: 얼굴은 나왔지만 랜드마크가 **실패**한 연속 프레임 수 (K와 비교).

---

## 6. ROI 경계 처리

명세에 ROI가 이미지 밖으로 나갔을 때 처리 방식은 없음.

- **구현**: `_clamp_roi()`로 `[x1, y1, x2, y2]`를 프레임 크기 `(width, height)` 안으로 클램프함.

---

## 7. Confidence 값

명세: 출력에 `confidence: float`만 명시.

- **구현**:  
  - Face Landmarker 랜드마크 기반으로 정상 mouth ROI를 쓸 때: `1.0`  
  - Fallback(얼굴 bbox 비율) 또는 이전 ROI 유지 시: `0.5`

---

## 8. 리소스·API

명세에 리소스 해제·API 형태는 없음.

- **구현**:  
  - `close()`: FaceLandmarker 해제, `debug=True`일 때 OpenCV 창 정리.  
  - `with MouthROITracker(...) as tracker:` 형태의 context manager 지원 (`__enter__` / `__exit__`).  
  - **API 호출**: 실행 중 외부 API/네트워크 호출 없음. 모델은 로컬 `face_landmarker.task` 파일만 사용.

---

## 9. 시각화 (Debug mode)

명세: “시각화 옵션 (debug mode)”만 언급.

- **구현**: `MouthROITracker(..., debug=True)`일 때  
  - 매 프레임 `_draw_debug()` 호출.  
  - 얼굴 bbox(녹색), mouth ROI(모드에 따라 색 구분), 모드 텍스트(“FACEMESH” / “FALLBACK”)를 `cv2.imshow("MouthROITracker (debug)", ...)`로 표시.

---

## 10. 단위 테스트 범위

명세: “Unit test 포함”만 언급.

- **구현**:  
  - `_clamp_roi`, `_area`, `TrackerState`, `OUTER_LIP_LANDMARK_INDICES` 등 순수 함수·상수.  
  - `MouthROITracker` 초기화·기본 파라미터.  
  - `process()` 출력 형식(mouth_roi 길이 4, roi_mode, confidence), ROI가 이미지 내부인지, fallback 비율(내부 `_fallback_roi_from_face` 비율) 등.  
  - 실제 영상이 아닌 더미 BGR 프레임으로 동작만 검증 (얼굴 없는 이미지 → FALLBACK/기본 ROI).

---

## 11. 추가 명세서(추가 명세서.md) 구현 가정

추가 명세서의 Gemini·GPT안을 조합해 구현했다. roi_valid: MouthROIResult에 추가, 얼굴 없을 때 False. FSM: IDLE→HAND_TO_MOUTH→CUP_TO_MOUTH→DRINKING→INGESTION_COMPLETE/FAIL. Hand/Cup overlap 30%, 연속 6/10프레임, State 2에서 Mouth ROI Lock. 타임아웃 손→컵 10초·전체 15초. Swallow cue 미구현. Hand: Hand Landmarker(hand_landmarker.task 선택). Cup: YOLOv8 bottle/cup. run_pill_ingestion.py로 파이프라인·--log JSONL. tests/test_ingestion_fsm.py.

### 11.1 실제 복용 시퀀스·좌우 무관

- **예시(오른손잡이)**: 화면 왼쪽에서 알약을 집어 입으로 가져와 먹은 뒤, 화면 우측에서 컵을 얼굴 우측으로 가져와 마시는 흐름.
- **반대(왼손잡이 등)**: 오른쪽에서 알약 → 입, 왼쪽에서 컵을 얼굴 왼쪽으로 가져오는 경우도 동일하게 인정해야 함.
- **구현**: FSM·Hand/Cup 검출은 **좌/우를 구분하지 않음**. 손 bbox·컵 bbox가 mouth ROI와 overlap 또는 거리 조건만 만족하면 “손→입”, “컵→입”으로 판정하므로, **양손·양쪽 모두 동일하게 지원**됨.

---

이 문서는 “명세에 없지만 구현에서 결정한 것”만 정리한 것이며, 명세와 충돌하는 부분이 있으면 명세를 우선합니다.

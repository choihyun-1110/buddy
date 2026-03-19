아래는 **Cursor에게 그대로 넘기면 되는 설계 명세서(.md 파일 내용)**야.
파일명 예시: mouth_roi_design_spec.md

⸻

Mouth ROI Tracking Module Design Spec

Project: Pill Ingestion Recognition
Goal: Robust mouth ROI extraction using FaceMesh + Fallback + Smoothing

⸻

1. Objective

알약 복용 인식 파이프라인에서 pill-to-mouth 이벤트 감지 정확도를 높이기 위해
안정적인 mouth ROI를 생성하는 모듈을 설계한다.

요구사항:
	•	입 위치를 프레임마다 정확히 추적
	•	손 가림(occlusion), landmark 튐 현상에 견고
	•	실시간 동작 가능 (>= 20 FPS 목표)
	•	ROI 좌표가 프레임 간 급격히 변하지 않도록 smoothing 적용

⸻

2. System Overview

Core Strategy

Primary:
	•	FaceMesh 기반 landmark → mouth ROI 생성

Fallback:
	•	FaceMesh confidence 낮거나 landmark 이상 시
	•	Face Detection 기반 비율 ROI 사용

Stabilization:
	•	ROI 좌표 EMA smoothing 적용

⸻

3. Processing Pipeline

Step 1. Face Detection
	•	얼굴 bbox 검출
	•	실패 시: 이전 프레임 ROI 유지 (최대 N 프레임)

Output:

face_bbox = [x_min, y_min, x_max, y_max]


⸻

Step 2. FaceMesh Landmark Inference

입 주변 landmark 추출

사용 landmark 그룹:
	•	Outer lip landmarks (기본)
	•	Optional: Inner lip landmarks (정밀 판단 시)

Landmark → pixel 좌표 변환

⸻

Step 3. Mouth ROI 생성 (Primary Path)

입 landmark 좌표 기반 bounding box 계산

xmin = min(x_i)
xmax = max(x_i)
ymin = min(y_i)
ymax = max(y_i)

Margin 추가:

mx = 0.15 * (xmax - xmin)
my = 0.20 * (ymax - ymin)

최종 ROI:

[xmin - mx,
 ymin - my,
 xmax + mx,
 ymax + my]


⸻

4. Fallback Strategy

FaceMesh 사용 불가 조건:
	•	landmark confidence < threshold
	•	landmark 좌표 급변 (이전 프레임 대비 2배 이상 이동)
	•	landmark가 face_bbox 외부 위치

Fallback 방식:

Face bbox 비율 기반 mouth ROI 생성

face_width  = face_bbox_width
face_height = face_bbox_height

mouth_roi:
x: 중앙 60%
y: 하단 40%

즉,

xmin = face_x + 0.2 * width
xmax = face_x + 0.8 * width
ymin = face_y + 0.6 * height
ymax = face_y + 1.0 * height


⸻

5. ROI Smoothing (EMA)

목적:
	•	ROI가 프레임마다 튀는 현상 방지
	•	pill event 경계 안정화

EMA 적용:

ROI_t = alpha * ROI_raw + (1 - alpha) * ROI_prev

권장 alpha:
	•	0.3 ~ 0.6
	•	초기값: 0.5

좌표별 독립 적용:

x1_t = alpha * x1_raw + (1-alpha) * x1_prev
y1_t = ...
x2_t = ...
y2_t = ...


⸻

6. Occlusion Handling

손이 입을 가릴 경우:

증상:
	•	landmark 튐
	•	ROI 갑자기 확장/축소

대응 전략:
	1.	sudden ROI area change > 2x → 무시
	2.	landmark confidence 낮을 때:
	•	최대 K 프레임까지 이전 ROI 유지
	3.	face detection은 유지되면 fallback ROI 사용

⸻

7. ROI State Machine

State 변수:

roi_valid
roi_mode = {FACEMESH, FALLBACK}
frames_since_valid

Logic:
	•	FaceMesh OK → FACEMESH 모드
	•	Fail 1~K frame → 이전 ROI 유지
	•	K frame 초과 → FALLBACK 모드
	•	FaceMesh 회복 → FACEMESH 복귀

⸻

8. Interface Specification

Input
	•	frame (BGR image)
	•	previous_roi
	•	previous_state

Output

{
    "mouth_roi": [x1, y1, x2, y2],
    "roi_mode": "FACEMESH" or "FALLBACK",
    "confidence": float
}


⸻

9. Performance Requirements
	•	Latency per frame < 20ms (640x480 기준)
	•	ROI drift < 10px per frame (정상 상황)
	•	Occlusion 후 1초 이내 복구

⸻

10. Integration with Pill Event Logic

이 mouth ROI는 다음 로직과 결합된다:
	•	pill bbox 중심이 mouth ROI 내부 진입
	•	이후 pill detection 사라짐
	•	hand가 mouth ROI에서 이탈

따라서 ROI는:
	•	과도하게 크면 안 됨 (false positive 증가)
	•	너무 작아도 안 됨 (미검출 증가)

⸻

11. Future Extensions
	•	Lip closure detection (FaceMesh lip distance 활용)
	•	Jaw motion velocity 기반 swallow cue
	•	Mouth depth estimation (if stereo or depth sensor available)

⸻

Final Deliverable Expectation

Cursor는 다음을 구현해야 한다:
	•	MouthROITracker 클래스
	•	내부:
	•	face detect
	•	facemesh inference
	•	fallback generator
	•	ema smoothing
	•	occlusion guard
	•	Unit test 포함
	•	시각화 옵션 (debug mode)

⸻

이 문서는 “복용 여부 인식의 안정성 확보”를 위한 핵심 ROI 모듈 명세다.
최종 ingestion 판단 로직은 별도 모듈에서 처리한다.

⸻

원하면 다음으로는
pill disappearance를 occlusion vs ingestion으로 구분하는 이벤트 설계 명세서도 만들어줄게.
이게 진짜 정확도 핵심이다.
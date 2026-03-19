# 복용 인식 프로세스 (Ingestion Process)

`run_pill_ingestion.py` 실행 시, 매 프레임마다 아래 순서로 인식·판정이 이뤄집니다.  
**추가요구사항.md** 반영: 오탐(false positive) 방지를 위해 보수적 임계값·시간창·컵 접촉 지속시간을 적용합니다.

---

## 1. 전체 흐름

```
[프레임] → Mouth ROI 트래커 → mouth_roi, roi_valid
        → Hand 트래커      → hand_bboxes[]
        → Cup 검출기       → cup_bbox, cup_conf (roi_valid일 때만 mouth_roi로 입 근처 컵 우선)
        → FSM.update()     → state, final_decision (O/X)
```

- **roi_valid=False** 인 프레임에서는 hand/cup 입 근처 판정을 **하지 않음** (이벤트 스킵, UNKNOWN 유지).
- **화면 고정 ROI**(얼굴 미검출 시 기본 영역)는 **디버그/표시용만** 쓰고, **판정 로직에는 사용하지 않음** (해당 시 roi_valid=False).

---

## 2. Mouth ROI (입 영역)

| 항목 | 설명 |
|------|------|
| **역할** | 얼굴에서 **입 주변 사각 영역 [x1,y1,x2,y2]** 를 구함. 손/컵 “입 근처”는 이 영역과 겹침·거리로 판단. |
| **구현** | `mouth_roi_tracker/tracker.py` — MediaPipe **Face Landmarker**, 로컬 `models/face_landmarker.task` |
| **출력** | `mouth_roi`, `roi_mode`(FACEMESH / FALLBACK), `roi_valid`, `confidence` |
| **roi_valid** | 얼굴·입이 제대로 잡힐 때만 True. **False가 연속 30프레임** 이상이면 FSM FAIL (occlusion 약 1초 허용). |
| **FALLBACK** | 얼굴 미검출 시: 이전 ROI HOLD(K프레임) 후 **roi_valid=False** 전환, 이벤트 판정 스킵. 화면 고정 ROI는 **판정에 사용 안 함**. |

---

## 3. Hand (손)

| 항목 | 설명 |
|------|------|
| **역할** | 손 bbox 목록. “손이 입에 갔다”는 mouth_roi와의 **overlap** 또는 **거리(mouth 대각선 비율)** 로 판단. |
| **구현** | `pill_ingestion/hand_tracker.py` — MediaPipe **Hand Landmarker**, 로컬 `models/hand_landmarker.task` |
| **입 근처 조건** | (1) mouth_roi와 bbox **overlap 비율 ≥ 0.05** **또는** (2) 손 중심이 mouth 대각선의 **0.8배 이내**. (1.2는 너무 넓어 오탐 가능 → 0.8) |

---

## 4. Cup (컵/병)

| 항목 | 설명 |
|------|------|
| **역할** | 컵/병 bbox 하나. “컵이 입에 갔다”는 mouth_roi와의 overlap 또는 **컵 중심–입 중심 거리(mouth 대각선 비율)** 로 판단. |
| **구현** | `pill_ingestion/cup_detector.py` — YOLO `yolov8n.pt`, COCO 39=bottle, 40=wine_glass, 41=cup. |
| **입 근처 조건** | overlap ≥ 0.15 **또는** 컵 중심이 mouth 대각선의 **0.8배 이내**. |
| **mouth_roi 전달** | **roi_valid일 때만** mouth_roi를 cup 검출기에 넘김. roi_valid=False일 때는 넘기지 않아 화면 고정 ROI로 컵 우선순위를 쓰지 않음. |
| **컵 미검출 허용** | 미검출이 **4프레임 연속**일 때만 카운트 리셋. 1~3프레임만 안 잡혀도 이전 카운트 유지. |

---

## 5. FSM (복용 시퀀스 판정)

구현: `pill_ingestion/ingestion_fsm.py`

### 5.1 상태 전이 (컵 사용 시, cup_required=True)

```
IDLE
  │  조건: roi_valid && 손이 입 근처인 프레임이 hand_min_frames(5)회 연속
  ▼
HAND_TO_MOUTH
  │  안전장치: 손이 입에서 한 번 떨어지는 이벤트 요구 (hand_leave_frames ≥ 3)
  │  조건: hand_left_after_to_mouth && 컵이 입 근처인 구간이 cup_min_frames(8)회
  ▼
CUP_TO_MOUTH
  │  조건: 컵이 입에 있는 구간이 cup_contact_duration_frames(15)회 = 약 0.5초
  ▼
INGESTION_COMPLETE (O)
```

- **손 → 입**: 손이 입 근처로 **5프레임 연속** 있어야 HAND_TO_MOUTH (1프레임 노이즈 방지).
- **손 이탈**: HAND_TO_MOUTH 이후 **손이 3프레임 이상 떨어진 뒤**에만 컵 이벤트 인정 (거짓 트리거 감소).
- **컵 → 입**: 컵이 입 근처로 **8프레임** 인정되면 CUP_TO_MOUTH.
- **O 조건**: “컵이 입에 가면 바로 O”가 아니라, **컵 접촉 지속 15프레임(약 0.5초)** 이상일 때만 O.

### 5.2 hand-only 모드 (--no-cup, cup_required=False)

- **디버그/대체 모드**로만 권장. 기본은 cup_required=True.
- 조건: 손이 입에 **15프레임(0.5초) 이상** 머문 뒤, 손이 **3프레임** 떨어지면 O.

### 5.3 실패(FAIL) 조건

- **roi_valid** 가 False인 구간이 **30프레임(약 1초)** 연속.
- HAND_TO_MOUTH 에서 **15초 안에** 컵이 입 근처로 오지 않음.
- **전체 15초** 타임아웃.
- CUP_TO_MOUTH 에서 컵이 4초 넘게 입에만 있음 (cup too long).

---

## 6. 주요 설정값 (IngestionConfig, 추가요구사항 반영)

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| hand_overlap_threshold | 0.05 | 손–입 overlap 최소 비율 (0.01은 스치기만 해도 True) |
| hand_near_mouth_max_ratio | 0.8 | 손 중심이 mouth 대각선의 N배 이내면 “입 근처” |
| hand_min_frames | 5 | 손이 입 근처인 연속 프레임 수 → HAND_TO_MOUTH |
| hand_leave_required_before_cup | 3 | 컵→입 인정 전, 손이 입에서 떨어진 최소 프레임 |
| cup_overlap_threshold | 0.15 | 컵–입 overlap 최소 비율 |
| cup_near_mouth_max_ratio | 0.8 | 컵 중심이 mouth 대각선의 N배 이내면 “입 근처” |
| cup_min_frames | 8 | 컵이 입 근처로 인정되는 구간 → CUP_TO_MOUTH |
| cup_contact_duration_frames | 15 | CUP_TO_MOUTH 후 컵 접촉 유지 프레임 수(약 0.5s) → O |
| max_consecutive_invalid_frames | 30 | roi_valid False 연속 이 수 초과 시 FAIL (occlusion 허용) |
| max_wait_hand_to_cup_sec | 15 | HAND_TO_MOUTH 후 컵 대기 시간(초) |
| hand_only_min_frames | 15 | hand-only 모드: 손이 입에 머무는 최소 프레임 |
| hand_leave_min_frames | 3 | hand-only: 손 이탈 연속 프레임 |

---

## 7. 실행 시 데이터 흐름 요약

1. **MouthROITracker.process(frame)**  
   → `mouth_roi`, `roi_valid`, `roi_mode`

2. **HandTracker.update(frame)**  
   → `hand_bboxes`

3. **CupDetector.update(frame, mouth_roi=mouth_roi if roi_valid else None)**  
   → `cup_bbox`, `cup_conf`. **roi_valid일 때만** mouth_roi로 입 근처 컵 우선.

4. **FSM.update(...)**  
   → roi_valid=False면 hand/cup 판정 스킵.  
   → `state`, `event_triggered`, `final_decision` (O/X).

5. **event_triggered && final_decision == "O"** 일 때만  
   로그 기록 및 `[복용 완료]` 한 번 출력.

좌/우(왼손/오른손, 컵 위치)는 구분하지 않고, **mouth_roi와의 겹침·거리만**으로 판단합니다.

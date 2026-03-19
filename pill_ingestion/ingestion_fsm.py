"""
복용 시퀀스 FSM (추가 명세: Gemini 4단계 + GPT roi_valid/연속프레임/타임아웃 조합).
Hand→Mouth → Cup→Mouth → Cup 지속 → Cup 이탈 시 복용 완료(O).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mouth_roi_tracker.types import IngestionState

# overlap: 두 bbox 교집합 면적 / mouth_roi 면적
def _overlap_ratio(roi: list[float], bbox: list[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x1 = max(roi[0], bbox[0])
    y1 = max(roi[1], bbox[1])
    x2 = min(roi[2], bbox[2])
    y2 = min(roi[3], bbox[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    if roi_area <= 0:
        return 0.0
    return inter / roi_area


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance_to_roi(roi: list[float], bbox: list[float]) -> float:
    """bbox 중심과 roi 중심 거리. roi 대각선 대비 비율(0에 가까울수록 가까움)."""
    cx, cy = _bbox_center(bbox)
    rx = (roi[0] + roi[2]) / 2
    ry = (roi[1] + roi[3]) / 2
    d = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
    diag = ((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2) ** 0.5
    if diag <= 0:
        return float("inf")
    return d / diag


@dataclass
class IngestionConfig:
    # 추가요구사항: 오탐 방지, 보수적 임계값
    hand_overlap_threshold: float = 0.05  # 0.01은 스치기만 해도 True → 0.05
    hand_near_mouth_max_ratio: float = 0.8  # 1.2는 너무 넓음 → 0.8 (mouth 대각선 기준)
    cup_overlap_threshold: float = 0.15
    cup_near_mouth_max_ratio: float = 0.8  # 1.2 → 0.8
    hand_min_frames: int = 5   # 1프레임 노이즈 방지 → 5 (약 0.17s @30fps)
    cup_min_frames: int = 8   # 2 → 8 (약 0.27s), 컵이 입에 '대는' 구간
    cup_contact_duration_frames: int = 15  # CUP_TO_MOUTH 후 컵 접촉 15프레임(0.5s) 이상 시 O
    min_drinking_sec: float = 0.5
    max_wait_hand_to_cup_sec: float = 15.0  # 10 → 15 (개인차 고려)
    max_cup_to_swallow_sec: float = 4.0
    overall_timeout_sec: float = 15.0
    drinking_fps: float = 30.0
    cup_required: bool = True
    hand_only_min_frames: int = 15   # hand-only 모드: 손이 입에 15프레임(0.5s) 이상 머물어야
    hand_leave_min_frames: int = 3   # HAND_TO_MOUTH 후 손이 3프레임 떨어져야 컵 이벤트 인정
    hand_leave_required_before_cup: int = 3  # 컵→입 인정 전 손 이탈 최소 프레임
    max_consecutive_invalid_frames: int = 30  # 5 → 30 (약 1초, occlusion 허용)


class IngestionSequenceDetector:
    """
    추가 명세서 조합:
    - Gemini: IDLE → HAND_TO_MOUTH → CUP_TO_MOUTH → INGESTION_COMPLETE, overlap 30%, min_drinking 1.5s
    - GPT: roi_valid 필수, 연속 프레임 확인, 타임아웃, Mouth ROI Lock at CUP_TO_MOUTH
    """
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.state = "IDLE"
        self._start_time_sec: float = 0.0
        self._hand_done_time_sec: Optional[float] = None
        self._cup_contact_time_sec: Optional[float] = None
        self._locked_mouth_roi: Optional[list[float]] = None  # State 2 진입 시 고정 (Gemini Occlusion)
        self._hand_overlap_frames = 0
        self._cup_overlap_frames = 0
        self._cup_at_mouth_frames = 0
        self._cup_missing_frames = 0  # 컵 미검출 연속 프레임 (이걸 넘기면 카운트 리셋)
        self._hand_leave_frames = 0
        self._event_triggered = False
        self._final_decision: Optional[str] = None
        self._ingestion_time_sec: Optional[float] = None
        self._consecutive_invalid: int = 0
        self._frames_in_hand_to_mouth: int = 0
        self._hand_left_after_to_mouth: bool = False  # HAND_TO_MOUTH 후 손이 한 번 떨어진 뒤에만 컵→O 인정

    def update(
        self,
        mouth_roi: list[float],
        roi_valid: bool,
        hand_bboxes: list[list[float]],
        cup_bbox: list[float],
        cup_conf: float,
        current_time_sec: float,
        swallow_event: bool = False,
        swallow_score: float = 0.0,
    ) -> IngestionState:
        if not roi_valid:
            self._consecutive_invalid += 1
            if self.state != "IDLE" and self._consecutive_invalid >= self.config.max_consecutive_invalid_frames:
                self.state = "FAIL"
                self._final_decision = "UNKNOWN"
                return IngestionState(
                    state=self.state,
                    event_triggered=False,
                    final_decision=self._final_decision,
                    message="roi invalid (consecutive)",
                    ingestion_time_sec=None,
                )
            # HAND_TO_MOUTH + hand-only: invalid = 손이 안 보임 → 손 뗀 걸로 간주하고 O 조건 확인
            cfg = self.config
            if self.state == "HAND_TO_MOUTH" and not cfg.cup_required:
                self._frames_in_hand_to_mouth += 1
                self._hand_leave_frames += 1
                if self._hand_leave_frames >= cfg.hand_leave_min_frames and self._frames_in_hand_to_mouth >= cfg.hand_only_min_frames:
                    self.state = "INGESTION_COMPLETE"
                    self._final_decision = "O"
                    self._ingestion_time_sec = current_time_sec - (self._hand_done_time_sec or 0)
                    return IngestionState(
                        state=self.state,
                        event_triggered=True,
                        final_decision=self._final_decision,
                        message="ingestion complete (hand only, roi invalid)",
                        ingestion_time_sec=self._ingestion_time_sec,
                    )
            return IngestionState(
                state=self.state,
                event_triggered=False,
                final_decision=self._final_decision,
                message=self.state,
                ingestion_time_sec=self._ingestion_time_sec,
            )
        self._consecutive_invalid = 0
        roi = self._locked_mouth_roi if self._locked_mouth_roi is not None else mouth_roi
        cfg = self.config
        event = False
        msg = ""

        # 전체 타임아웃
        if self.state != "IDLE" and (current_time_sec - self._start_time_sec) > cfg.overall_timeout_sec:
            self.state = "FAIL"
            self._final_decision = "X"
            return IngestionState(state=self.state, event_triggered=False, final_decision="X", message="timeout", ingestion_time_sec=None)

        # Hand–mouth: overlap 또는 손 중심이 입 근처(거리 비율)
        hand_overlap = False
        for hb in hand_bboxes:
            if _overlap_ratio(roi, hb) >= cfg.hand_overlap_threshold:
                hand_overlap = True
                break
            if _distance_to_roi(roi, hb) <= cfg.hand_near_mouth_max_ratio:
                hand_overlap = True
                break
        if hand_overlap:
            self._hand_overlap_frames += 1
            self._hand_leave_frames = 0
        else:
            self._hand_overlap_frames = 0
            if self.state == "HAND_TO_MOUTH":
                self._hand_leave_frames += 1
            else:
                self._hand_leave_frames = 0

        # Cup–mouth: overlap 또는 컵 중심이 입 근처(거리 비율). 컵이 1~2프레임만 잡혀도 유지되도록 미검출 허용.
        cup_overlap = False
        if cup_bbox and len(cup_bbox) >= 4:
            if _overlap_ratio(roi, cup_bbox) >= cfg.cup_overlap_threshold:
                cup_overlap = True
            elif _distance_to_roi(roi, cup_bbox) <= cfg.cup_near_mouth_max_ratio:
                cup_overlap = True
        if cup_overlap:
            self._cup_missing_frames = 0
            self._cup_overlap_frames += 1
            if self.state in ("CUP_TO_MOUTH", "DRINKING"):
                self._cup_at_mouth_frames += 1
        else:
            self._cup_missing_frames += 1
            if self._cup_missing_frames > 4:  # 4프레임 연속 미검출 시에만 리셋 (잠깐 안 잡혀도 유지)
                self._cup_overlap_frames = 0
                if self.state in ("CUP_TO_MOUTH", "DRINKING"):
                    self._cup_at_mouth_frames = 0

        # FSM transitions
        if self.state == "IDLE":
            if self._hand_overlap_frames >= cfg.hand_min_frames:
                self.state = "HAND_TO_MOUTH"
                self._start_time_sec = current_time_sec
                self._hand_done_time_sec = current_time_sec
                self._frames_in_hand_to_mouth = 0
                event = True
                msg = "hand to mouth"

        elif self.state == "HAND_TO_MOUTH":
            self._frames_in_hand_to_mouth += 1
            if (current_time_sec - (self._hand_done_time_sec or 0)) > cfg.max_wait_hand_to_cup_sec:
                self.state = "FAIL"
                self._final_decision = "X"
                return IngestionState(state=self.state, event_triggered=False, final_decision="X", message="timeout hand→cup", ingestion_time_sec=None)
            if self._hand_leave_frames >= getattr(cfg, "hand_leave_required_before_cup", 3):
                self._hand_left_after_to_mouth = True
            if self._cup_overlap_frames >= cfg.cup_min_frames and self._hand_left_after_to_mouth:
                self._locked_mouth_roi = list(roi)
                self.state = "CUP_TO_MOUTH"
                self._cup_contact_time_sec = current_time_sec
                self._cup_at_mouth_frames = self._cup_overlap_frames
                event = True
                msg = "cup to mouth"
            elif not cfg.cup_required and self._hand_leave_frames >= cfg.hand_leave_min_frames:
                if self._frames_in_hand_to_mouth >= cfg.hand_only_min_frames:
                    self.state = "INGESTION_COMPLETE"
                    self._final_decision = "O"
                    self._ingestion_time_sec = current_time_sec - (self._hand_done_time_sec or 0)
                    event = True
                    msg = "ingestion complete (hand only)"

        elif self.state == "CUP_TO_MOUTH":
            # 삼킴 감지 시 즉시 완료 (컵 접촉 프레임 조건 생략)
            need_frames = getattr(cfg, "cup_contact_duration_frames", 15)
            min_cup_before_swallow = max(4, need_frames // 3)  # 최소 컵 접촉 후 swallow 인정
            swallow_triggered = (
                swallow_event and self._cup_at_mouth_frames >= min_cup_before_swallow
            )
            if swallow_triggered or self._cup_at_mouth_frames >= need_frames:
                self.state = "INGESTION_COMPLETE"
                self._final_decision = "O"
                self._ingestion_time_sec = current_time_sec - (self._hand_done_time_sec or 0)
                event = True
                msg = "ingestion complete (swallow)" if swallow_triggered else "ingestion complete"
            elif self._cup_missing_frames > 4 and self._cup_overlap_frames == 0:
                self._cup_at_mouth_frames = 0
            elif (current_time_sec - (self._cup_contact_time_sec or 0)) > cfg.max_cup_to_swallow_sec:
                self.state = "FAIL"
                self._final_decision = "X"
                return IngestionState(state=self.state, event_triggered=False, final_decision="X", message="cup too long", ingestion_time_sec=None)

        return IngestionState(
            state=self.state,
            event_triggered=event,
            final_decision=self._final_decision,
            message=msg or self.state,
            ingestion_time_sec=self._ingestion_time_sec,
        )

    def reset(self) -> None:
        self.state = "IDLE"
        self._hand_done_time_sec = None
        self._cup_contact_time_sec = None
        self._locked_mouth_roi = None
        self._hand_overlap_frames = 0
        self._cup_overlap_frames = 0
        self._cup_at_mouth_frames = 0
        self._cup_missing_frames = 0
        self._hand_leave_frames = 0
        self._event_triggered = False
        self._final_decision = None
        self._ingestion_time_sec = None
        self._consecutive_invalid = 0
        self._frames_in_hand_to_mouth = 0
        self._hand_left_after_to_mouth = False

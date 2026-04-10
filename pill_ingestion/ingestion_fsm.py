"""
Pill ingestion sequence FSM.
IDLE → HAND_TO_MOUTH → CUP_TO_MOUTH → VERIFY → INGESTION_COMPLETE
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mouth_roi_tracker.types import IngestionState


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
    return inter / roi_area if roi_area > 0 else 0.0


def _distance_ratio(roi: list[float], bbox: list[float]) -> float:
    """bbox 중심 ↔ roi 중심 거리 / roi 대각선 길이."""
    if not bbox or len(bbox) < 4:
        return float("inf")
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rx = (roi[0] + roi[2]) / 2
    ry = (roi[1] + roi[3]) / 2
    d = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
    diag = ((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2) ** 0.5
    return d / diag if diag > 0 else float("inf")


@dataclass
class IngestionConfig:
    # Hand → Mouth
    hand_overlap_threshold: float = 0.05
    hand_near_mouth_max_ratio: float = 0.8
    hand_min_frames: int = 3

    # Cup → Mouth
    cup_overlap_threshold: float = 0.05
    cup_near_mouth_max_ratio: float = 1.5
    cup_min_frames: int = 2
    cup_contact_duration_frames: int = 8
    cup_missing_tolerance: int = 6   # 컵이 N프레임 안 보여도 유지

    # Timeouts
    max_wait_hand_to_cup_sec: float = 30.0
    max_cup_to_swallow_sec: float = 25.0
    max_verify_sec: float = 15.0
    overall_timeout_sec: float = 60.0

    # ROI invalid 허용 프레임
    max_consecutive_invalid_frames: int = 60

    # Hand-only mode (cup_required=False)
    cup_required: bool = True
    hand_only_min_frames: int = 15
    hand_leave_min_frames: int = 1


class IngestionSequenceDetector:
    """
    IDLE → HAND_TO_MOUTH → CUP_TO_MOUTH → VERIFY → INGESTION_COMPLETE

    VERIFY: 컵이 입에서 떨어진 후 입을 크게 벌리면 완료 확정.
            타임아웃 시에도 컵 접촉이 충분했으면 완료로 인정.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.state = "IDLE"
        self._start_time: float = 0.0
        self._hand_done_time: Optional[float] = None
        self._cup_contact_time: Optional[float] = None
        self._verify_start_time: Optional[float] = None
        self._locked_mouth_roi: Optional[list[float]] = None

        self._hand_overlap_frames: int = 0
        self._hand_leave_frames: int = 0
        self._frames_in_hand_to_mouth: int = 0

        self._cup_overlap_frames: int = 0
        self._cup_at_mouth_frames: int = 0
        self._cup_peak_frames: int = 0    # 리셋 전 최대값 보존
        self._cup_missing_frames: int = 0

        self._consecutive_invalid: int = 0
        self._final_decision: Optional[str] = None
        self._ingestion_time: Optional[float] = None

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
        mouth_open_event: bool = False,
    ) -> IngestionState:

        # ROI invalid 처리
        if not roi_valid:
            self._consecutive_invalid += 1
            if self.state != "IDLE" and self._consecutive_invalid >= self.config.max_consecutive_invalid_frames:
                return self._fail("roi invalid too long", "UNKNOWN")
            return IngestionState(
                state=self.state, event_triggered=False,
                final_decision=self._final_decision,
                message=self.state, ingestion_time_sec=self._ingestion_time,
            )

        self._consecutive_invalid = 0
        cfg = self.config
        roi = self._locked_mouth_roi if self._locked_mouth_roi is not None else mouth_roi

        # 전체 타임아웃
        if self.state != "IDLE" and (current_time_sec - self._start_time) > cfg.overall_timeout_sec:
            return self._fail("overall timeout", "X")

        # 손-입 근접 계산
        hand_near = any(
            _overlap_ratio(roi, hb) >= cfg.hand_overlap_threshold
            or _distance_ratio(roi, hb) <= cfg.hand_near_mouth_max_ratio
            for hb in hand_bboxes
        )
        if hand_near:
            self._hand_overlap_frames += 1
            self._hand_leave_frames = 0
        else:
            self._hand_overlap_frames = 0
            if self.state == "HAND_TO_MOUTH":
                self._hand_leave_frames += 1

        # 컵-입 근접 계산
        cup_near = bool(cup_bbox) and (
            _overlap_ratio(roi, cup_bbox) >= cfg.cup_overlap_threshold
            or _distance_ratio(roi, cup_bbox) <= cfg.cup_near_mouth_max_ratio
        )
        if cup_near:
            self._cup_missing_frames = 0
            self._cup_overlap_frames += 1
            if self.state == "CUP_TO_MOUTH":
                self._cup_at_mouth_frames += 1
                self._cup_peak_frames = max(self._cup_peak_frames, self._cup_at_mouth_frames)
        else:
            self._cup_missing_frames += 1
            if self._cup_missing_frames > cfg.cup_missing_tolerance:
                self._cup_overlap_frames = 0
                if self.state == "CUP_TO_MOUTH":
                    self._cup_at_mouth_frames = 0

        # ── FSM 전이 ─────────────────────────────────────────────────

        if self.state == "IDLE":
            if self._hand_overlap_frames >= cfg.hand_min_frames:
                self.state = "HAND_TO_MOUTH"
                self._start_time = current_time_sec
                self._hand_done_time = current_time_sec
                self._frames_in_hand_to_mouth = 0
                return self._event("hand to mouth")

        elif self.state == "HAND_TO_MOUTH":
            self._frames_in_hand_to_mouth += 1

            if (current_time_sec - (self._hand_done_time or 0)) > cfg.max_wait_hand_to_cup_sec:
                return self._fail("timeout hand→cup", "X")

            if self._cup_overlap_frames >= cfg.cup_min_frames:
                self._locked_mouth_roi = list(roi)
                self.state = "CUP_TO_MOUTH"
                self._cup_contact_time = current_time_sec
                self._cup_at_mouth_frames = self._cup_overlap_frames
                return self._event("cup to mouth")

            if not cfg.cup_required and self._hand_leave_frames >= cfg.hand_leave_min_frames:
                if self._frames_in_hand_to_mouth >= cfg.hand_only_min_frames:
                    self._final_decision = "O"
                    self._ingestion_time = current_time_sec - (self._hand_done_time or 0)
                    self.state = "INGESTION_COMPLETE"
                    return self._event("ingestion complete (hand only)")

        elif self.state == "CUP_TO_MOUTH":
            need = cfg.cup_contact_duration_frames
            min_before = max(3, need // 3)
            cup_timeout = (current_time_sec - (self._cup_contact_time or 0)) > cfg.max_cup_to_swallow_sec
            cup_enough = self._cup_peak_frames >= need   # 리셋 전 최대값으로 판정
            cup_left = self._cup_missing_frames > cfg.cup_missing_tolerance and self._cup_overlap_frames == 0

            if swallow_event and self._cup_peak_frames >= min_before:
                self.state = "VERIFY"
                self._verify_start_time = current_time_sec
                return self._event("swallow detected → verify")

            if cup_enough and cup_left:
                self.state = "VERIFY"
                self._verify_start_time = current_time_sec
                return self._event("cup left mouth → verify")

            if cup_timeout:
                if self._cup_peak_frames >= min_before:
                    self.state = "VERIFY"
                    self._verify_start_time = current_time_sec
                    return self._event("cup timeout → verify")
                return self._fail("cup timeout no contact", "X")

        elif self.state == "VERIFY":
            elapsed = current_time_sec - (self._verify_start_time or current_time_sec)

            if mouth_open_event:
                self._final_decision = "O"
                self._ingestion_time = current_time_sec - (self._hand_done_time or 0)
                self.state = "INGESTION_COMPLETE"
                return self._event("mouth open verified → complete")

            if elapsed > cfg.max_verify_sec:
                self._final_decision = "O"
                self._ingestion_time = current_time_sec - (self._hand_done_time or 0)
                self.state = "INGESTION_COMPLETE"
                return self._event("verify timeout → complete (assumed)")

        return IngestionState(
            state=self.state, event_triggered=False,
            final_decision=self._final_decision,
            message=self.state, ingestion_time_sec=self._ingestion_time,
        )

    def _event(self, msg: str) -> IngestionState:
        return IngestionState(
            state=self.state, event_triggered=True,
            final_decision=self._final_decision,
            message=msg, ingestion_time_sec=self._ingestion_time,
        )

    def _fail(self, msg: str, decision: str) -> IngestionState:
        self.state = "FAIL"
        self._final_decision = decision
        return IngestionState(
            state="FAIL", event_triggered=False,
            final_decision=decision,
            message=msg, ingestion_time_sec=None,
        )

    def reset(self) -> None:
        self.__init__(self.config)

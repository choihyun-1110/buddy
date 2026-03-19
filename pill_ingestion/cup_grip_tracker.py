"""
컵 쥐기 상태 추적기.

YOLO 컵 bbox + 손 그리핑 상태를 결합해
"어떤 손이 컵을 쥐고 있는지" 프레임 단위로 추적한다.

손에 가려서 YOLO가 컵을 잃어도 hold_timeout_frames 동안
마지막 컵 위치를 유지한 채 held=True 를 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pill_ingestion.hand_visualizer import bbox_overlap_ratio, detect_grip


@dataclass
class CupHoldState:
    """한 프레임의 컵 쥐기 결과."""
    cup_bbox: list[float]        # 현재 컵 bbox (추정 포함)
    held: bool                   # 어떤 손이 쥐고 있는지 여부
    held_by: int                 # 쥔 손 인덱스 (-1 = 없음)
    yolo_visible: bool           # YOLO가 이번 프레임에서 컵을 실제로 본 건지
    hold_frames: int             # 연속 hold 프레임 수


class CupGripTracker:
    """
    매 프레임 update()를 호출해 컵 쥐기 상태를 추적.

    Parameters
    ----------
    hold_timeout_frames:
        YOLO가 컵을 잃어도 hold 상태를 유지할 최대 프레임 수.
        기본 20 (30fps 기준 약 0.67s).
    overlap_threshold:
        손 bbox ∩ 컵 bbox 비율이 이 값 이상이면 "쥐고 있다"로 판정.
    min_grip_frames:
        컵을 처음 인식할 때 최소 연속 프레임 수 (노이즈 방지).
    """

    def __init__(
        self,
        hold_timeout_frames: int = 20,
        overlap_threshold: float = 0.25,
        min_grip_frames: int = 3,
    ):
        self._timeout = hold_timeout_frames
        self._overlap_th = overlap_threshold
        self._min_grip = min_grip_frames

        self._last_cup_bbox: list[float] = []
        self._miss_count: int = 0          # YOLO가 컵을 못 본 연속 프레임
        self._held_by: int = -1
        self._hold_frames: int = 0
        self._candidate_frames: int = 0    # min_grip_frames 확인용

    def reset(self) -> None:
        self._last_cup_bbox = []
        self._miss_count = 0
        self._held_by = -1
        self._hold_frames = 0
        self._candidate_frames = 0

    def update(
        self,
        cup_bbox_yolo: list[float],        # CupDetector.update() 결과 (없으면 [])
        hand_bboxes: list[list[float]],    # 각 손 bbox
        hand_landmarks_list: list[list[list[float]]],  # 각 손 랜드마크
    ) -> CupHoldState:
        """
        컵 쥐기 상태를 갱신하고 현재 상태를 반환.
        """
        yolo_found = len(cup_bbox_yolo) == 4

        # 1) 각 손의 그리핑 상태 계산
        gripping_flags: list[bool] = []
        for lms in hand_landmarks_list:
            g, _ = detect_grip(lms)
            gripping_flags.append(g)

        # 2) YOLO가 컵을 찾은 경우 → 어느 손이 쥐고 있는지 확인
        if yolo_found:
            self._last_cup_bbox = cup_bbox_yolo
            self._miss_count = 0

            best_hand = -1
            best_ratio = 0.0
            for i, (hbbox, gripping) in enumerate(zip(hand_bboxes, gripping_flags)):
                if not gripping:
                    continue
                ratio = bbox_overlap_ratio(hbbox, cup_bbox_yolo)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_hand = i

            if best_hand >= 0 and best_ratio >= self._overlap_th:
                # 후보 프레임 누적 (노이즈 방지)
                self._candidate_frames += 1
                if self._candidate_frames >= self._min_grip:
                    self._held_by = best_hand
                    self._hold_frames += 1
                    return CupHoldState(
                        cup_bbox=cup_bbox_yolo,
                        held=True,
                        held_by=best_hand,
                        yolo_visible=True,
                        hold_frames=self._hold_frames,
                    )
            else:
                # 그리핑 손 없거나 겹침 부족 → candidate 리셋
                self._candidate_frames = 0
                if self._held_by == -1:
                    self._hold_frames = 0
                    return CupHoldState(
                        cup_bbox=cup_bbox_yolo,
                        held=False,
                        held_by=-1,
                        yolo_visible=True,
                        hold_frames=0,
                    )

        # 3) YOLO가 컵을 못 찾은 경우
        else:
            self._miss_count += 1

        # 4) 이전에 hold 중이었고 timeout 미만이면 → 마지막 위치 유지
        if self._held_by >= 0 and self._miss_count <= self._timeout:
            # 쥔 손이 여전히 그리핑 중인지 확인
            still_gripping = (
                self._held_by < len(gripping_flags)
                and gripping_flags[self._held_by]
            )
            if still_gripping:
                self._hold_frames += 1
                return CupHoldState(
                    cup_bbox=self._last_cup_bbox,
                    held=True,
                    held_by=self._held_by,
                    yolo_visible=yolo_found,
                    hold_frames=self._hold_frames,
                )

        # 5) hold 해제
        self._held_by = -1
        self._hold_frames = 0
        self._candidate_frames = 0
        return CupHoldState(
            cup_bbox=self._last_cup_bbox if yolo_found else [],
            held=False,
            held_by=-1,
            yolo_visible=yolo_found,
            hold_frames=0,
        )

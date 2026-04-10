"""
삼킴 + 입 열림 감지 (추가 명세 2.4).

FaceMesh 내측 입술 landmark(13=상단, 14=하단) 거리 변화로
삼킴 신호(swallow_event)와 입 크게 열림(mouth_open_event)을 산출한다.

mouth_open_event:
    알약 복용 후 "입 벌려서 보여주기" 단계 감지.
    VERIFY 상태에서 INGESTION_COMPLETE 트리거로 사용.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwallowResult:
    swallow_score: float      # 0~1, 1에 가까울수록 입술 닫힘
    swallow_event: bool       # 삼킴 감지 (입술 닫힘 연속)
    lip_gap_ratio: float      # 입 너비 대비 상하 거리 (디버그용)
    mouth_open_event: bool    # 입 크게 열림 감지 (VERIFY 단계용)


class SwallowEstimator:
    """
    Parameters
    ----------
    ema_alpha:
        lip_gap_ratio EMA smoothing 계수.
    close_ratio:
        이 값 이하 → 닫힘 판정 (삼킴).
    open_ratio:
        이 값 이상 → 크게 열림 판정 (복용 확인).
    min_close_frames:
        연속 닫힘 프레임 수 → swallow_event.
    min_open_frames:
        연속 열림 프레임 수 → mouth_open_event.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        close_ratio: float = 0.18,
        score_k: float = 3.0,
        min_close_frames: int = 4,
        open_ratio: float = 0.40,
        min_open_frames: int = 5,
    ):
        self._alpha = ema_alpha
        self._close_ratio = close_ratio
        self._score_k = score_k
        self._min_close = min_close_frames
        self._open_ratio = open_ratio
        self._min_open = min_open_frames
        self._smoothed_gap: float | None = None
        self._close_frames: int = 0
        self._open_frames: int = 0

    def reset(self) -> None:
        """CUP_TO_MOUTH 진입 시 호출."""
        self._smoothed_gap = None
        self._close_frames = 0
        self._open_frames = 0

    def update(self, lip_landmarks: list[list[float]]) -> SwallowResult:
        """
        lip_landmarks:
            [[upper_inner_x,y], [lower_inner_x,y],
             [left_corner_x,y], [right_corner_x,y]]
        """
        if len(lip_landmarks) < 4:
            return SwallowResult(
                swallow_score=0.0, swallow_event=False,
                lip_gap_ratio=0.0, mouth_open_event=False,
            )

        ux, uy = lip_landmarks[0][:2]
        lx, ly = lip_landmarks[1][:2]
        cx_l, cy_l = lip_landmarks[2][:2]
        cx_r, cy_r = lip_landmarks[3][:2]

        gap = ((ux - lx) ** 2 + (uy - ly) ** 2) ** 0.5
        mouth_width = ((cx_r - cx_l) ** 2 + (cy_r - cy_l) ** 2) ** 0.5

        if mouth_width < 1e-3:
            return SwallowResult(
                swallow_score=0.0, swallow_event=False,
                lip_gap_ratio=0.0, mouth_open_event=False,
            )

        gap_ratio = gap / mouth_width

        if self._smoothed_gap is None:
            self._smoothed_gap = gap_ratio
        else:
            self._smoothed_gap = (
                self._alpha * gap_ratio + (1 - self._alpha) * self._smoothed_gap
            )

        # 삼킴 감지 (입 닫힘)
        raw_score = max(0.0, 1.0 - self._score_k * (self._smoothed_gap / max(self._close_ratio, 1e-6)))
        score = min(1.0, raw_score)

        if self._smoothed_gap <= self._close_ratio:
            self._close_frames += 1
        else:
            self._close_frames = 0

        # 입 크게 열림 감지 (복용 확인)
        if self._smoothed_gap >= self._open_ratio:
            self._open_frames += 1
        else:
            self._open_frames = 0

        return SwallowResult(
            swallow_score=score,
            swallow_event=self._close_frames >= self._min_close,
            lip_gap_ratio=round(self._smoothed_gap, 4),
            mouth_open_event=self._open_frames >= self._min_open,
        )

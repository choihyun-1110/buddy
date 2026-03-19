"""
삼킴 감지 (추가 명세 2.4).

FaceMesh 내측 입술 landmark(13=상단, 14=하단) 거리 변화로
삼킴 신호(swallow_score)를 산출한다.

사용법:
    estimator = SwallowEstimator()
    estimator.reset()          # CUP_TO_MOUTH 진입 시 반드시 호출

    result = estimator.update(lip_landmarks)  # 매 프레임

핵심 아이디어:
    - 컵을 든 직후엔 입술이 약간 열림 (lip_gap 양수)
    - 삼키는 순간 입술이 닫힘 → lip_gap 감소
    - lip_gap_ratio (입 너비 대비 상하 거리) 가 close_ratio 이하로
      min_close_frames 연속 유지되면 swallow_event = True
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwallowResult:
    swallow_score: float   # 0~1, 1에 가까울수록 입술 닫힘
    swallow_event: bool    # score >= threshold 가 min_close_frames 연속 유지됨
    lip_gap_ratio: float   # 입 너비 대비 상하 거리 (디버그용)


class SwallowEstimator:
    """
    Parameters
    ----------
    ema_alpha:
        lip_gap_ratio EMA smoothing 계수 (0=완전 smooth, 1=raw).
    close_ratio:
        입 너비 대비 상하 거리가 이 값 이하이면 "닫힘" 판정.
        0.18 = 입 너비의 18% 이하.
    score_k:
        score 계산 기울기. 클수록 gap 변화에 민감.
    min_close_frames:
        연속 닫힘 프레임이 이 수 이상이면 swallow_event = True.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        close_ratio: float = 0.18,
        score_k: float = 3.0,
        min_close_frames: int = 4,
    ):
        self._alpha = ema_alpha
        self._close_ratio = close_ratio
        self._score_k = score_k
        self._min_close = min_close_frames
        self._smoothed_gap: float | None = None
        self._close_frames: int = 0

    def reset(self) -> None:
        """CUP_TO_MOUTH 진입 시 호출해 이전 상태 초기화."""
        self._smoothed_gap = None
        self._close_frames = 0

    def update(self, lip_landmarks: list[list[float]]) -> SwallowResult:
        """
        Parameters
        ----------
        lip_landmarks:
            MouthROIResult["lip_landmarks"] 그대로 전달.
            [[upper_inner_x, upper_inner_y],
             [lower_inner_x, lower_inner_y],
             [left_corner_x, left_corner_y],
             [right_corner_x, right_corner_y]]
            비어 있으면 score=0 반환.
        """
        if len(lip_landmarks) < 4:
            return SwallowResult(swallow_score=0.0, swallow_event=False, lip_gap_ratio=0.0)

        ux, uy = lip_landmarks[0][:2]
        lx, ly = lip_landmarks[1][:2]
        cx_l, cy_l = lip_landmarks[2][:2]
        cx_r, cy_r = lip_landmarks[3][:2]

        gap = ((ux - lx) ** 2 + (uy - ly) ** 2) ** 0.5
        mouth_width = ((cx_r - cx_l) ** 2 + (cy_r - cy_l) ** 2) ** 0.5

        if mouth_width < 1e-3:
            return SwallowResult(swallow_score=0.0, swallow_event=False, lip_gap_ratio=0.0)

        gap_ratio = gap / mouth_width

        if self._smoothed_gap is None:
            self._smoothed_gap = gap_ratio
        else:
            self._smoothed_gap = (
                self._alpha * gap_ratio + (1 - self._alpha) * self._smoothed_gap
            )

        raw_score = max(0.0, 1.0 - self._score_k * (self._smoothed_gap / max(self._close_ratio, 1e-6)))
        score = min(1.0, raw_score)

        if self._smoothed_gap <= self._close_ratio:
            self._close_frames += 1
        else:
            self._close_frames = 0

        event = self._close_frames >= self._min_close

        return SwallowResult(
            swallow_score=score,
            swallow_event=event,
            lip_gap_ratio=round(self._smoothed_gap, 4),
        )

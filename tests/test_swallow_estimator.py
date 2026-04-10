"""SwallowEstimator 단위 테스트."""
import pytest
from pill_ingestion.swallow_estimator import SwallowEstimator


def _landmarks(gap_ratio: float, width: float = 100.0):
    """gap_ratio = upper-lower / mouth_width 인 더미 랜드마크."""
    gap = gap_ratio * width
    return [
        [50.0, 0.0],           # upper inner
        [50.0, gap],           # lower inner
        [0.0,  gap / 2],       # left corner
        [width, gap / 2],      # right corner
    ]


def test_returns_zero_on_empty():
    est = SwallowEstimator()
    res = est.update([])
    assert res.swallow_score == 0.0
    assert res.swallow_event is False
    assert res.mouth_open_event is False


def test_swallow_event_on_closed_mouth():
    est = SwallowEstimator(min_close_frames=3)
    est.reset()
    for _ in range(5):
        res = est.update(_landmarks(0.05))   # 닫힌 상태
    assert res.swallow_event is True
    assert res.swallow_score >= 0.0


def test_mouth_open_event():
    est = SwallowEstimator(min_open_frames=3)
    est.reset()
    for _ in range(5):
        res = est.update(_landmarks(0.55))   # 크게 열린 상태
    assert res.mouth_open_event is True


def test_no_event_on_normal_gap():
    est = SwallowEstimator(min_close_frames=4, min_open_frames=4)
    est.reset()
    for _ in range(10):
        res = est.update(_landmarks(0.28))   # 보통 상태
    assert res.swallow_event is False
    assert res.mouth_open_event is False


def test_reset_clears_state():
    est = SwallowEstimator(min_close_frames=3)
    for _ in range(5):
        est.update(_landmarks(0.05))
    est.reset()
    res = est.update(_landmarks(0.05))
    assert res.swallow_event is False   # reset 후 첫 프레임은 아직 미충족

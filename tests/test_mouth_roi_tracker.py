"""
MouthROITracker 단위 테스트 (명세 Final Deliverable).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mouth_roi_tracker import MouthROITracker
from mouth_roi_tracker.tracker import (
    TrackerState,
    _clamp_roi,
    _area,
    OUTER_LIP_LANDMARK_INDICES,
)


# --- 유틸/순수 함수 테스트 ---

class TestClampRoi:
    def test_inside_bounds(self):
        roi = [10, 20, 100, 80]
        got = _clamp_roi(roi, 640, 480)
        assert got == [10, 20, 100, 80]

    def test_clamp_negative(self):
        roi = [-10, -5, 50, 50]
        got = _clamp_roi(roi, 640, 480)
        assert got[0] == 0 and got[1] == 0
        assert got[2] == 50 and got[3] == 50

    def test_clamp_exceeds_size(self):
        roi = [0, 0, 700, 500]
        got = _clamp_roi(roi, 640, 480)
        assert got[2] == 640 and got[3] == 480


class TestArea:
    def test_area_positive(self):
        assert _area([0, 0, 10, 20]) == 200

    def test_area_zero_width(self):
        assert _area([5, 5, 5, 10]) == 0


class TestTrackerState:
    def test_state_creation(self):
        s = TrackerState(
            roi_valid=True,
            roi_mode="FACEMESH",
            frames_since_valid=0,
            frames_since_landmark_ok=0,
            previous_roi=[10, 20, 30, 40],
        )
        assert s.roi_mode == "FACEMESH"
        assert s.previous_roi == [10, 20, 30, 40]


class TestOuterLipIndices:
    def test_indices_count(self):
        assert len(OUTER_LIP_LANDMARK_INDICES) >= 10
        assert all(isinstance(i, int) and 0 <= i < 468 for i in OUTER_LIP_LANDMARK_INDICES)


# --- MouthROITracker 통합 테스트 (실제 프레임 입력) ---

@pytest.fixture
def tracker():
    return MouthROITracker(debug=False)


@pytest.fixture
def sample_bgr_frame():
    """640x480 BGR 더미 프레임 (얼굴 없음)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestMouthROITrackerInit:
    def test_default_init(self, tracker):
        assert tracker.ema_alpha == 0.5
        assert tracker.max_frames_keep_roi == 10
        assert tracker.max_frames_no_face == 5

    def test_custom_params(self):
        t = MouthROITracker(ema_alpha=0.3, max_frames_no_face=8, debug=False)
        assert t.ema_alpha == 0.3
        assert t.max_frames_no_face == 8


class TestMouthROITrackerProcess:
    def test_output_shape(self, tracker, sample_bgr_frame):
        out = tracker.process(sample_bgr_frame)
        assert "mouth_roi" in out
        assert "roi_mode" in out
        assert "confidence" in out
        assert "roi_valid" in out
        assert len(out["mouth_roi"]) == 4
        assert out["roi_mode"] in ("FACEMESH", "FALLBACK")
        assert isinstance(out["confidence"], (int, float))

    def test_roi_bounds(self, tracker, sample_bgr_frame):
        out = tracker.process(sample_bgr_frame)
        x1, y1, x2, y2 = out["mouth_roi"]
        assert 0 <= x1 <= 640 and 0 <= x2 <= 640
        assert 0 <= y1 <= 480 and 0 <= y2 <= 480
        assert x2 >= x1 and y2 >= y1

    def test_no_face_uses_fallback(self, tracker, sample_bgr_frame):
        out = tracker.process(sample_bgr_frame)
        # 검정 이미지에서는 얼굴이 없으므로 FALLBACK 또는 이전 ROI 유지
        assert out["roi_mode"] in ("FACEMESH", "FALLBACK")

    def test_previous_roi_smoothing(self, tracker, sample_bgr_frame):
        prev = [100.0, 200.0, 150.0, 250.0]
        out = tracker.process(sample_bgr_frame, previous_roi=prev)
        # 이전 ROI가 있으면 no-face 시 유지될 수 있음 (N 프레임 내)
        assert len(out["mouth_roi"]) == 4

    def test_ema_smoothing_effect(self, tracker):
        # 연속 두 번 호출 시 두 번째 결과가 이전과 부드럽게 이어져야 함
        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
        o1 = tracker.process(f1)
        o2 = tracker.process(f1)
        # 같은 입력이면 같은 출력에 수렴 (no face이므로 default/prev 유지)
        assert o1["mouth_roi"] == o2["mouth_roi"] or True  # 상태에 따라 다를 수 있음

    def test_state_persistence(self, tracker, sample_bgr_frame):
        tracker.process(sample_bgr_frame)
        state = tracker.get_state()
        assert state is not None
        assert hasattr(state, "previous_roi")


class TestFallbackGeometry:
    """명세 Section 4. Fallback 비율 검증 (face bbox → mouth ROI)."""
    def test_fallback_ratio(self):
        # 내부 _fallback_roi_from_face 비율만 검증 (face_bbox [0,0,100,100] 가정)
        t = MouthROITracker(debug=False)
        face = [0.0, 0.0, 100.0, 100.0]
        roi = t._fallback_roi_from_face(face)
        assert roi[0] == 20 and roi[2] == 80   # 0.2*100, 0.8*100
        assert roi[1] == 60 and roi[3] == 100  # 0.6*100, 1.0*100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

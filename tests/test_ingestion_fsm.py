"""
복용 시퀀스 FSM 단위 테스트 (추가 명세서).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pill_ingestion.ingestion_fsm import IngestionSequenceDetector, IngestionConfig


def _roi():
    return [100, 200, 180, 260]


def _hand_overlapping_roi():
    return [110, 210, 170, 250]


def _cup_overlapping_roi():
    return [120, 215, 165, 255]


def test_fsm_idle_without_roi_valid():
    cfg = IngestionConfig(hand_min_frames=2, cup_min_frames=2, max_wait_hand_to_cup_sec=5, min_drinking_sec=0.5, drinking_fps=30)
    fsm = IngestionSequenceDetector(config=cfg)
    out = fsm.update(mouth_roi=_roi(), roi_valid=False, hand_bboxes=[_hand_overlapping_roi()], cup_bbox=[], cup_conf=0, current_time_sec=0)
    assert out["state"] in ("IDLE", "FAIL")
    assert out["final_decision"] in (None, "UNKNOWN")


def test_fsm_hand_to_mouth_then_cup():
    cfg = IngestionConfig(
        hand_min_frames=2,
        cup_min_frames=2,
        hand_leave_required_before_cup=3,
        cup_contact_duration_frames=15,
        max_wait_hand_to_cup_sec=10,
    )
    fsm = IngestionSequenceDetector(config=cfg)
    roi = _roi()
    hand = _hand_overlapping_roi()
    cup = _cup_overlapping_roi()
    t = 0.0
    # Hand overlap 2 frames → HAND_TO_MOUTH
    for _ in range(2):
        out = fsm.update(roi, True, [hand], [], 0, t)
        t += 0.033
    assert out["state"] == "HAND_TO_MOUTH"
    # 손이 3프레임 떨어짐 (hand_left_after_to_mouth)
    for _ in range(3):
        out = fsm.update(roi, True, [], [], 0, t)
        t += 0.033
    # Cup 2 frames → CUP_TO_MOUTH
    for _ in range(2):
        out = fsm.update(roi, True, [], cup, 0.9, t)
        t += 0.033
    assert out["state"] == "CUP_TO_MOUTH"
    # 컵 접촉 15프레임 → INGESTION_COMPLETE (O)
    for _ in range(15):
        out = fsm.update(roi, True, [], cup, 0.9, t)
        t += 0.033
    assert out["state"] == "INGESTION_COMPLETE"
    assert out["final_decision"] == "O"
    assert out.get("ingestion_time_sec") is not None


def test_fsm_timeout_hand_to_cup():
    cfg = IngestionConfig(hand_min_frames=2, cup_min_frames=2, max_wait_hand_to_cup_sec=0.2, drinking_fps=30)
    fsm = IngestionSequenceDetector(config=cfg)
    roi = _roi()
    hand = _hand_overlapping_roi()
    out = fsm.update(roi, True, [hand], [], 0, 0.0)
    out = fsm.update(roi, True, [hand], [], 0, 0.033)
    assert out["state"] == "HAND_TO_MOUTH"
    # Wait > 0.2s without cup
    out = fsm.update(roi, True, [], [], 0, 0.5)
    assert out["state"] == "FAIL"
    assert out["final_decision"] == "X"

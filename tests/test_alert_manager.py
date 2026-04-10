"""AlertManager 단위 테스트."""
import pytest
from pill_ingestion.alert_manager import AlertManager


def test_no_alert_below_threshold():
    mgr = AlertManager(patient_id="test", webhook_url="", threshold=5)
    state = mgr.on_fsm_event("FAIL", "HAND_TO_MOUTH", 10.0, "timeout")
    assert state.score == 2
    assert state.alert_sent is False


def test_alert_fires_at_threshold():
    mgr = AlertManager(patient_id="test", webhook_url="", threshold=5)
    mgr.on_fsm_event("FAIL", "HAND_TO_MOUTH", 5.0, "timeout")   # +2
    mgr.on_fsm_event("FAIL", "HAND_TO_MOUTH", 10.0, "timeout")  # +3 (2회 연속)
    state = mgr._state
    assert state.score >= 5
    assert state.alert_sent is True


def test_verify_skipped_adds_score():
    mgr = AlertManager(patient_id="test", webhook_url="", threshold=99)
    state = mgr.on_fsm_event("INGESTION_COMPLETE", "VERIFY", 20.0, "verify timeout → complete (assumed)")
    assert state.score == 3


def test_reset_clears_score():
    mgr = AlertManager(patient_id="test", webhook_url="", threshold=5)
    mgr.on_fsm_event("FAIL", "HAND_TO_MOUTH", 5.0, "timeout")
    mgr.reset_session()
    assert mgr._state.score == 0
    assert mgr._state.alert_sent is False


def test_face_hidden_adds_score():
    mgr = AlertManager(patient_id="test", webhook_url="", threshold=99)
    mgr.on_face_hidden(time_sec=3.0, duration_frames=65)
    assert mgr._state.score == 2
    assert len(mgr._state.events) == 1

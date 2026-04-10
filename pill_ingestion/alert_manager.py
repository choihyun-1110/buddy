"""
Suspicion tracking + Slack webhook alert.

Usage:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    export PATIENT_ID="room-101-kim"   # optional

    alerter = AlertManager(patient_id="room-101-kim")
    alerter.on_fsm_event(state, prev_state, time_sec)
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SuspicionEvent:
    time_sec: float
    reason: str
    score_delta: int


@dataclass
class SuspicionState:
    score: int = 0
    events: list[SuspicionEvent] = field(default_factory=list)
    alert_sent: bool = False


# 의심 패턴별 점수
_SCORE = {
    "fail_timeout":     2,   # 타임아웃 FAIL (약 안 먹었을 가능성)
    "face_hidden":      2,   # 얼굴 장시간 가림
    "verify_skipped":   3,   # VERIFY 없이 완료 (입 안 보여줌)
    "repeated_fail":    3,   # 연속 FAIL (여러 번 시도)
}

_ALERT_THRESHOLD = 5   # 이 점수 이상이면 슬랙 전송


class AlertManager:
    """
    FSM 이벤트를 받아 의심 점수를 누적하고,
    임계값 초과 시 Slack webhook으로 알림 전송.

    Parameters
    ----------
    patient_id:
        환자 식별자 (CLI --patient-id 또는 env PATIENT_ID).
    webhook_url:
        Slack incoming webhook URL (env SLACK_WEBHOOK_URL).
    threshold:
        알림 발송 최소 점수.
    """

    def __init__(
        self,
        patient_id: str = "unknown",
        webhook_url: Optional[str] = None,
        threshold: int = _ALERT_THRESHOLD,
    ):
        self.patient_id = patient_id or os.getenv("PATIENT_ID", "unknown")
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        self.threshold = threshold
        self._state = SuspicionState()
        self._consecutive_fails: int = 0
        self._session_start = datetime.now()

    def on_fsm_event(
        self,
        state: str,
        prev_state: str,
        time_sec: float,
        message: str = "",
    ) -> SuspicionState:
        """
        FSM 상태 전이 시 호출.
        의심 점수를 갱신하고 필요 시 알림 발송.
        """
        reason = None
        delta = 0

        # FAIL 진입
        if state == "FAIL" and prev_state != "FAIL":
            self._consecutive_fails += 1
            if self._consecutive_fails >= 2:
                reason = f"repeated_fail ({self._consecutive_fails}회 연속 실패)"
                delta = _SCORE["repeated_fail"]
            else:
                reason = f"fail_timeout: {message}"
                delta = _SCORE["fail_timeout"]

        # INGESTION_COMPLETE 진입 — verify 없이 타임아웃으로 넘어온 경우
        elif state == "INGESTION_COMPLETE" and "assumed" in message:
            reason = "verify_skipped: 입 확인 없이 완료 처리됨"
            delta = _SCORE["verify_skipped"]

        # 정상 완료면 consecutive_fails 리셋
        if state == "INGESTION_COMPLETE" and "assumed" not in message:
            self._consecutive_fails = 0

        if reason and delta > 0:
            self._state.score += delta
            self._state.events.append(SuspicionEvent(
                time_sec=round(time_sec, 1),
                reason=reason,
                score_delta=delta,
            ))
            self._maybe_alert(time_sec)

        return self._state

    def on_face_hidden(self, time_sec: float, duration_frames: int) -> None:
        """roi_valid=False 가 오래 지속될 때 호출."""
        reason = f"face_hidden: {duration_frames}프레임 얼굴 가림"
        self._state.score += _SCORE["face_hidden"]
        self._state.events.append(SuspicionEvent(
            time_sec=round(time_sec, 1),
            reason=reason,
            score_delta=_SCORE["face_hidden"],
        ))
        self._maybe_alert(time_sec)

    def reset_session(self) -> None:
        """새 복약 세션 시작 시 호출 (FSM reset과 함께)."""
        self._state = SuspicionState()
        self._consecutive_fails = 0
        self._session_start = datetime.now()

    def _maybe_alert(self, time_sec: float) -> None:
        if self._state.alert_sent:
            return
        if self._state.score < self.threshold:
            return
        self._state.alert_sent = True
        self._send_slack(time_sec)

    def _send_slack(self, time_sec: float) -> None:
        if not self.webhook_url:
            print(f"[AlertManager] SLACK_WEBHOOK_URL not set. Alert suppressed.")
            print(f"[AlertManager] Patient: {self.patient_id} | Score: {self._state.score}")
            for ev in self._state.events:
                print(f"  t={ev.time_sec}s  +{ev.score_delta}  {ev.reason}")
            return

        event_lines = "\n".join(
            f"• t={ev.time_sec}s  (+{ev.score_delta}) {ev.reason}"
            for ev in self._state.events
        )

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "⚠️ 복약 의심 패턴 감지",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*환자 ID*\n{self.patient_id}"},
                        {"type": "mrkdwn", "text": f"*의심 점수*\n{self._state.score}점"},
                        {"type": "mrkdwn", "text": f"*세션 시작*\n{self._session_start.strftime('%H:%M:%S')}"},
                        {"type": "mrkdwn", "text": f"*감지 시각*\n{time_sec:.1f}s"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*이벤트 목록*\n{event_lines}",
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "직접 확인 필요"},
                            "style": "danger",
                            "value": self.patient_id,
                        }
                    ],
                },
            ]
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[AlertManager] Slack alert sent. Patient: {self.patient_id}, Score: {self._state.score}")
                else:
                    print(f"[AlertManager] Slack response: {resp.status}")
        except urllib.error.URLError as e:
            print(f"[AlertManager] Slack send failed: {e}")

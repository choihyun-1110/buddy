#!/usr/bin/env python3
"""
복용 시퀀스 파이프라인 (추가 명세서): Mouth ROI + Hand + Cup + FSM.
상태를 화면 상단에 표시, 완료 시 로그 저장.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2

from mouth_roi_tracker import MouthROITracker
from mouth_roi_tracker.types import MouthROIResult, IngestionState
from pill_ingestion import HandTracker, CupDetector, IngestionSequenceDetector
from pill_ingestion.ingestion_fsm import IngestionConfig
from pill_ingestion.hand_visualizer import draw_all_hands
from pill_ingestion.cup_grip_tracker import CupGripTracker, CupHoldState
from pill_ingestion.swallow_estimator import SwallowEstimator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "dataset" / "taking_phill" / "IMG_0963.MOV"


def draw_overlay(
    frame,
    mouth_roi: list[float],
    hand_out: dict,
    cup_bbox: list,
    state: IngestionState,
    roi_mode: str | None = None,
) -> None:
    """mouth_roi, 손 AR 뼈대, cup_bbox, state 텍스트 표기."""
    # 손 AR 뼈대 (랜드마크 있을 때)
    if hand_out.get("hand_landmarks"):
        draw_all_hands(frame, hand_out["hand_landmarks"], hand_out.get("handedness", []))
    else:
        # 랜드마크 없으면 bbox만 (폴백)
        for hb in hand_out.get("hand_bboxes", []):
            if len(hb) == 4:
                x1, y1, x2, y2 = [int(x) for x in hb]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    # Mouth ROI (얼굴 검출 시에만; FALLBACK이면 생략)
    if mouth_roi and len(mouth_roi) == 4 and roi_mode != "FALLBACK":
        x1, y1, x2, y2 = [int(x) for x in mouth_roi]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # Cup
    if cup_bbox and len(cup_bbox) == 4:
        x1, y1, x2, y2 = [int(x) for x in cup_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, "CUP", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    # State (상단)
    state_text = state.get("state", "IDLE")
    if state.get("final_decision"):
        state_text += f" | {state['final_decision']}"
    cv2.putText(
        frame, state_text, (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
    )
    if state.get("message"):
        cv2.putText(
            frame, state["message"][:40], (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )


def main():
    parser = argparse.ArgumentParser(description="복용 시퀀스 파이프라인 (Hand→Mouth→Cup)")
    parser.add_argument("--video", nargs="?", const=str(DEFAULT_VIDEO), default=None, metavar="PATH")
    parser.add_argument("--no-debug", action="store_true", help="mouth ROI만 표시, hand/cup/state 끄기")
    parser.add_argument("--log", type=str, default="", help="JSONL 로그 저장 경로 (예: logs/events.jsonl)")
    parser.add_argument("--no-cup", action="store_true", help="컵 검출 없이 손→입→손 떠남만으로 복용 성공 인정 (컵이 안 잡힐 때)")
    parser.add_argument("--cup-conf", type=float, default=None, metavar="0.0~1.0", help="컵 검출 최소 신뢰도 (기본 0.10). 컵이 안 잡히면 0.05 등으로 낮춰 보세요.")
    args = parser.parse_args()

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            print(f"영상 파일을 찾을 수 없습니다: {video_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"영상을 열 수 없습니다: {video_path}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"비디오: {video_path.name}, FPS: {fps:.1f}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = 30
    delay = int(1000 / fps) if fps > 0 else 33
    paused = False
    time_sec = 0.0

    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
    else:
        log_file = None

    mouth_tracker = MouthROITracker(debug=False)
    hand_tracker = HandTracker()
    cup_kw = {} if args.cup_conf is None else {"conf_threshold": args.cup_conf}
    cup_detector = CupDetector(**cup_kw)
    cup_grip = CupGripTracker()
    swallow_est = SwallowEstimator()
    if not args.no_cup and not cup_detector.is_loaded():
        err = getattr(cup_detector, "_load_error", "알 수 없음")
        print("컵 검출 비활성: 모델 로드 실패.", file=sys.stderr)
        print(f"  원인: {err}", file=sys.stderr)
        print("  조치: 1) pip install ultralytics  2) models/README.md 참고해 models/yolov8n.pt 받아 두기.", file=sys.stderr)
    elif not args.no_cup:
        print("컵 검출: 사용 가능 (YOLO 로드됨)")
    fsm_config = IngestionConfig(cup_required=not args.no_cup)
    fsm = IngestionSequenceDetector(config=fsm_config)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time_sec = 0.0
                        fsm.reset()
                        continue
                    break
                mouth_out: MouthROIResult = mouth_tracker.process(frame)
                hand_out = hand_tracker.update(frame)
                # roi_valid일 때만 mouth_roi 전달 (화면 고정 ROI로 컵 우선순위 쓰지 않음)
                cup_out = cup_detector.update(
                    frame,
                    mouth_roi=mouth_out.get("mouth_roi") if mouth_out.get("roi_valid") else None,
                )
                grip_state: CupHoldState = cup_grip.update(
                    cup_bbox_yolo=cup_out.get("cup_bbox") or [],
                    hand_bboxes=hand_out["hand_bboxes"],
                    hand_landmarks_list=hand_out["hand_landmarks"],
                )
                # 삼킴 감지: CUP_TO_MOUTH 진입 시 reset, 그 외엔 매 프레임 업데이트
                prev_fsm_state = fsm.state
                swallow_res = swallow_est.update(mouth_out.get("lip_landmarks") or [])
                # FSM에는 "손이 실제로 쥔 컵" bbox만 전달 (held=True 일 때만 유효한 bbox 사용)
                fsm_cup_bbox = grip_state.cup_bbox if grip_state.held else []
                state = fsm.update(
                    mouth_roi=mouth_out["mouth_roi"],
                    roi_valid=mouth_out.get("roi_valid", True),
                    hand_bboxes=hand_out["hand_bboxes"],
                    cup_bbox=fsm_cup_bbox,
                    cup_conf=cup_out.get("cup_conf") or 0.0,
                    current_time_sec=time_sec,
                    swallow_event=swallow_res.swallow_event,
                    swallow_score=swallow_res.swallow_score,
                )
                # CUP_TO_MOUTH 진입 시 SwallowEstimator 리셋
                if prev_fsm_state != "CUP_TO_MOUTH" and fsm.state == "CUP_TO_MOUTH":
                    swallow_est.reset()
                time_sec += 1.0 / fps

                if not args.no_debug:
                    draw_overlay(
                        frame,
                        mouth_out["mouth_roi"],
                        hand_out,
                        grip_state.cup_bbox,  # hold 중이면 마지막 위치 유지, 아니면 YOLO bbox
                        state,
                        roi_mode=mouth_out.get("roi_mode"),
                    )
                    # 컵 쥐기 상태 표시
                    if grip_state.held:
                        cv2.putText(frame, f"HOLDING cup (hand {grip_state.held_by})",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                    if fsm.state == "CUP_TO_MOUTH":
                        sw_color = (0, 255, 0) if swallow_res.swallow_event else (180, 180, 180)
                        cv2.putText(frame, f"swallow score: {swallow_res.swallow_score:.2f}  gap: {swallow_res.lip_gap_ratio:.3f}",
                                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sw_color, 1, cv2.LINE_AA)
                    if state.get("event_triggered") and log_file:
                        log_file.write(json.dumps({
                            "t": time_sec,
                            "state": state["state"],
                            "message": state.get("message", ""),
                            "final_decision": state.get("final_decision"),
                            "ingestion_time_sec": state.get("ingestion_time_sec"),
                            "cup_conf": round(cup_out.get("cup_conf") or 0, 3),
                            "cup_detected": bool(cup_out.get("cup_bbox")),
                        }, ensure_ascii=False) + "\n")
                        log_file.flush()
                    # O로 전환된 그 순간에만 한 번만 출력 (이후 프레임에서는 state가 계속 O라서 event_triggered일 때만)
                    if state.get("event_triggered") and state.get("final_decision") == "O":
                        print(f"[복용 완료] 소요 시간: {state.get('ingestion_time_sec', 0):.1f}s")

            cv2.imshow("Pill Ingestion", frame)
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                fsm.reset()
                time_sec = 0.0
                print("FSM 리셋")
    finally:
        mouth_tracker.close()
        hand_tracker.close()
        if log_file:
            log_file.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

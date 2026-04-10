#!/usr/bin/env python3
"""Pill ingestion pipeline: Mouth ROI + Hand + Cup + FSM."""
import argparse
import json
import sys
from pathlib import Path

import cv2

from mouth_roi_tracker import MouthROITracker
from mouth_roi_tracker.types import MouthROIResult, IngestionState
from pill_ingestion import HandTracker, CupDetector, IngestionSequenceDetector
from pill_ingestion.ingestion_fsm import IngestionConfig
from pill_ingestion.hand_visualizer import draw_all_hands
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
    swallow_score: float = 0.0,
    lip_gap: float = 0.0,
    mouth_open_event: bool = False,
) -> None:
    # Hand skeleton
    if hand_out.get("hand_landmarks"):
        draw_all_hands(frame, hand_out["hand_landmarks"], hand_out.get("handedness", []))
    else:
        for hb in hand_out.get("hand_bboxes", []):
            if len(hb) == 4:
                x1, y1, x2, y2 = [int(v) for v in hb]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    # Mouth ROI (FACEMESH일 때만)
    if mouth_roi and len(mouth_roi) == 4 and roi_mode != "FALLBACK":
        x1, y1, x2, y2 = [int(v) for v in mouth_roi]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Cup
    if cup_bbox and len(cup_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in cup_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, "CUP", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    # FSM state
    state_text = state.get("state", "IDLE")
    if state.get("final_decision"):
        state_text += f" | {state['final_decision']}"
    cv2.putText(frame, state_text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    fsm_state = state.get("state", "IDLE")

    # CUP_TO_MOUTH: swallow score
    if fsm_state == "CUP_TO_MOUTH":
        color = (0, 255, 0) if swallow_score > 0.5 else (180, 180, 180)
        cv2.putText(frame, f"swallow: {swallow_score:.2f}  gap: {lip_gap:.3f}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # VERIFY: 입 열기 안내
    if fsm_state == "VERIFY":
        color = (0, 255, 0) if mouth_open_event else (0, 200, 255)
        cv2.putText(frame, "VERIFY: Open mouth wide (Ah~)",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"gap: {lip_gap:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if state.get("message") and fsm_state not in ("CUP_TO_MOUTH", "VERIFY"):
        cv2.putText(frame, state["message"][:60], (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Pill ingestion pipeline (Hand->Mouth->Cup)")
    parser.add_argument("--video", nargs="?", const=str(DEFAULT_VIDEO), default=None, metavar="PATH")
    parser.add_argument("--no-debug", action="store_true", help="Hide overlay")
    parser.add_argument("--log", type=str, default="", metavar="PATH", help="JSONL log path")
    parser.add_argument("--no-cup", action="store_true", help="Hand-only mode (no cup required)")
    parser.add_argument("--cup-conf", type=float, default=None, metavar="FLOAT",
                        help="Cup detection confidence threshold (default 0.05)")
    parser.add_argument("--save", type=str, default="", metavar="PATH", help="Save output video")
    args = parser.parse_args()

    # ── 카메라/영상 열기 ─────────────────────────────────────────
    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            print(f"Video not found: {video_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video: {video_path.name}, FPS: {fps:.1f}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam.", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = 30

    delay = max(1, int(1000 / fps))
    time_sec = 0.0
    paused = False

    # ── 로그 파일 ────────────────────────────────────────────────
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")

    # ── 영상 저장 ────────────────────────────────────────────────
    writer = None
    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = PROJECT_ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Saving to: {save_path}")

    # ── 컴포넌트 초기화 ──────────────────────────────────────────
    mouth_tracker = MouthROITracker(debug=False)
    hand_tracker = HandTracker()
    cup_kw = {} if args.cup_conf is None else {"conf_threshold": args.cup_conf}
    cup_detector = CupDetector(**cup_kw)
    swallow_est = SwallowEstimator()

    if not args.no_cup:
        if cup_detector.is_loaded():
            print("Cup detection: ready")
        else:
            print(f"Cup detection disabled: {getattr(cup_detector, '_load_error', 'unknown')}", file=sys.stderr)

    fsm = IngestionSequenceDetector(config=IngestionConfig(cup_required=not args.no_cup))

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
                cup_out = cup_detector.update(
                    frame,
                    mouth_roi=mouth_out.get("mouth_roi") if mouth_out.get("roi_valid") else None,
                )

                prev_state = fsm.state
                swallow_res = swallow_est.update(mouth_out.get("lip_landmarks") or [])

                state = fsm.update(
                    mouth_roi=mouth_out["mouth_roi"],
                    roi_valid=mouth_out.get("roi_valid", True),
                    hand_bboxes=hand_out["hand_bboxes"],
                    cup_bbox=cup_out.get("cup_bbox") or [],
                    cup_conf=cup_out.get("cup_conf") or 0.0,
                    current_time_sec=time_sec,
                    swallow_event=swallow_res.swallow_event,
                    swallow_score=swallow_res.swallow_score,
                    mouth_open_event=swallow_res.mouth_open_event,
                )

                # CUP_TO_MOUTH 진입 시 SwallowEstimator 리셋
                if prev_state != "CUP_TO_MOUTH" and fsm.state == "CUP_TO_MOUTH":
                    swallow_est.reset()

                time_sec += 1.0 / fps

                if not args.no_debug:
                    draw_overlay(
                        frame,
                        mouth_out["mouth_roi"],
                        hand_out,
                        cup_out.get("cup_bbox") or [],
                        state,
                        roi_mode=mouth_out.get("roi_mode"),
                        swallow_score=swallow_res.swallow_score,
                        lip_gap=swallow_res.lip_gap_ratio,
                        mouth_open_event=swallow_res.mouth_open_event,
                    )

                if state.get("event_triggered"):
                    if log_file:
                        log_file.write(json.dumps({
                            "t": round(time_sec, 3),
                            "state": state["state"],
                            "message": state.get("message", ""),
                            "final_decision": state.get("final_decision"),
                            "ingestion_time_sec": state.get("ingestion_time_sec"),
                            "cup_conf": round(cup_out.get("cup_conf") or 0, 3),
                        }, ensure_ascii=False) + "\n")
                        log_file.flush()

                    if state.get("final_decision") == "O":
                        print(f"[Ingestion complete] elapsed: {state.get('ingestion_time_sec', 0):.1f}s")

            if writer:
                writer.write(frame)
            cv2.imshow("Pill Ingestion", frame)
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                fsm.reset()
                time_sec = 0.0
                print("FSM reset")

    finally:
        mouth_tracker.close()
        hand_tracker.close()
        if log_file:
            log_file.close()
        if writer:
            writer.release()
            print("Video saved.")
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

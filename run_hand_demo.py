#!/usr/bin/env python3
"""
손 랜드마크 AR 시각화 데모.

웹캠 또는 영상 파일에서 손을 실시간 감지하고
MediaPipe 21개 keypoint 뼈대 + 그리핑 + 컵 쥐기 상태를 표시한다.

사용:
    python run_hand_demo.py              # 웹캠
    python run_hand_demo.py --video PATH # 영상 파일
    python run_hand_demo.py --no-cup     # 컵 감지 없이 손만

조작:
    q     종료
    space 일시정지/재개
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pill_ingestion.hand_tracker import HandTracker
from pill_ingestion.hand_visualizer import draw_all_hands
from pill_ingestion.cup_detector import CupDetector
from pill_ingestion.cup_grip_tracker import CupGripTracker

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "dataset" / "taking_phill" / "IMG_0963.MOV"


def draw_cup(frame: np.ndarray, cup_bbox: list[float], held: bool, yolo_visible: bool) -> None:
    """컵 bbox 그리기. held=True이면 초록, YOLO 추정 위치이면 점선 스타일."""
    if not cup_bbox or len(cup_bbox) < 4:
        return
    x1, y1, x2, y2 = [int(x) for x in cup_bbox]
    if held:
        color = (0, 220, 80)   # 초록: 손에 쥐고 있음
        thickness = 2
        label = "CUP (held)"
    else:
        color = (0, 140, 255)  # 주황: 탐지됨, 쥐지 않음
        thickness = 2
        label = "CUP"

    if not yolo_visible and held:
        # YOLO가 못 본 상태 → 점선으로 추정 위치 표시
        _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
        label = "CUP (hidden)"
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _draw_dashed_rect(frame, pt1, pt2, color, thickness=1, gap=8):
    """점선 사각형."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + gap, y2)), color, thickness)


def draw_hud(
    frame: np.ndarray,
    hand_count: int,
    gripping_list: list[bool],
    handedness_list: list[str],
    cup_held: bool,
    fps_display: float,
) -> None:
    h, w = frame.shape[:2]

    # 반투명 상단 배너
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(banner, 0.55, frame, 0.45, 0, frame)

    # 손 수
    cv2.putText(frame, f"Hands: {hand_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # 그리핑 상태
    grip_parts = []
    for gripping, side in zip(gripping_list, handedness_list):
        label = side[0]
        grip_parts.append(f"{label}:{'GRIP' if gripping else 'open'}")
    grip_text = "  ".join(grip_parts) if grip_parts else "-"
    cv2.putText(frame, grip_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    # 컵 쥐기 상태
    cup_text = "CUP: HELD" if cup_held else "CUP: -"
    cup_color = (0, 220, 80) if cup_held else (120, 120, 120)
    cv2.putText(frame, cup_text, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cup_color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps_display:.1f} FPS", (w - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="손 랜드마크 + 컵 쥐기 AR 데모")
    parser.add_argument("--video", nargs="?", const=str(DEFAULT_VIDEO), default=None,
                        metavar="PATH", help="영상 파일 경로 (생략 시 웹캠)")
    parser.add_argument("--no-cup", action="store_true", help="컵 감지 비활성")
    parser.add_argument("--cup-conf", type=float, default=0.05,
                        help="컵 YOLO 최소 신뢰도 (기본 0.05)")
    args = parser.parse_args()

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            print(f"파일 없음: {video_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"비디오: {video_path.name}  FPS: {fps:.1f}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = 30.0
        print("웹캠 모드")

    if not cap.isOpened():
        print("캡처 장치를 열 수 없습니다.", file=sys.stderr)
        return 1

    delay_ms = max(1, int(1000 / fps))
    hand_tracker = HandTracker()

    cup_detector = None
    cup_grip_tracker = None
    if not args.no_cup:
        cup_detector = CupDetector(conf_threshold=args.cup_conf)
        if cup_detector.is_loaded():
            cup_grip_tracker = CupGripTracker()
            print("컵 감지: 활성 (YOLO)")
        else:
            print("컵 감지: 비활성 (YOLO 로드 실패)")
            cup_detector = None

    paused = False
    prev_time = time.perf_counter()
    fps_display = fps
    print("q: 종료  space: 일시정지")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # 손 감지
                hand_out = hand_tracker.update(frame)

                # 컵 감지 + 쥐기 판정
                cup_hold = None
                if cup_detector is not None and cup_grip_tracker is not None:
                    cup_result = cup_detector.update(frame)
                    cup_hold = cup_grip_tracker.update(
                        cup_bbox_yolo=cup_result.get("cup_bbox") or [],
                        hand_bboxes=hand_out["hand_bboxes"],
                        hand_landmarks_list=hand_out["hand_landmarks"],
                    )

                # 컵 bbox 그리기 (손 뼈대보다 먼저 → 뼈대 위에 컵 라벨이 안 가려짐)
                if cup_hold is not None:
                    draw_cup(frame, cup_hold.cup_bbox, cup_hold.held, cup_hold.yolo_visible)

                # 손 AR 뼈대
                cup_held_flags = None
                if cup_hold is not None and cup_hold.held:
                    cup_held_flags = [
                        i == cup_hold.held_by
                        for i in range(hand_out["hand_count"])
                    ]
                gripping_list = draw_all_hands(
                    frame,
                    hand_out["hand_landmarks"],
                    hand_out["handedness"],
                    cup_held_flags=cup_held_flags,
                )

                # FPS 계산
                now = time.perf_counter()
                elapsed = now - prev_time
                if elapsed > 0:
                    fps_display = 0.9 * fps_display + 0.1 * (1.0 / elapsed)
                prev_time = now

                draw_hud(
                    frame,
                    hand_count=hand_out["hand_count"],
                    gripping_list=gripping_list,
                    handedness_list=hand_out["handedness"],
                    cup_held=cup_hold.held if cup_hold else False,
                    fps_display=fps_display,
                )

            cv2.imshow("Hand + Cup Grip Demo", frame)
            key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

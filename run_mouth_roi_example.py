#!/usr/bin/env python3
"""
Mouth ROI Tracker 예제 실행 (웹캠 / 비디오 파일).
사용법:
  python3 run_mouth_roi_example.py                    # 웹캠 실시간 (debug=True)
  python3 run_mouth_roi_example.py --video            # 테스트 영상 (dataset/taking_phill/IMG_0963.MOV)
  python3 run_mouth_roi_example.py --video path.MOV   # 지정한 비디오 파일
"""
import argparse
import sys
from pathlib import Path

import cv2

from mouth_roi_tracker import MouthROITracker

# 스크립트 기준 프로젝트 루트 (기본 테스트 영상 경로 계산용)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_VIDEO = PROJECT_ROOT / "dataset" / "taking_phill" / "IMG_0963.MOV"


def main():
    parser = argparse.ArgumentParser(description="Mouth ROI Tracker 예제")
    parser.add_argument(
        "--video",
        nargs="?",
        const=str(DEFAULT_TEST_VIDEO),
        default=None,
        metavar="PATH",
        help="비디오 파일로 실행 (경로 생략 시 dataset/taking_phill/IMG_0963.MOV)",
    )
    parser.add_argument("--no-debug", action="store_true", help="debug 시각화 끄기")
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"비디오 재생: {video_path.name} (프레임 수: {total_frames}, FPS: {fps:.1f})")
        print("종료: q, 일시정지: 스페이스")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = 30
        print("웹캠 재생 중. 종료: q")

    delay = int(1000 / fps) if fps > 0 else 33
    paused = False

    with MouthROITracker(debug=not args.no_debug) as tracker:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                out = tracker.process(frame)
                if args.no_debug:
                    print(out["roi_mode"], out["mouth_roi"])

            if not args.no_debug:
                key = cv2.waitKey(delay if not paused else 0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = not paused

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())

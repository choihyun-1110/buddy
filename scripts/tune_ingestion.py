#!/usr/bin/env python3
"""
영상 1회 재생 → 프레임별 데이터 수집 → 여러 FSM 파라미터로 시뮬레이션 → O 나오는 설정 찾기.
헤드리스(imshow 없음). 약 복용 영상에서 final_decision O가 나올 때까지 파인튜닝.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mouth_roi_tracker import MouthROITracker
from mouth_roi_tracker.types import MouthROIResult
from pill_ingestion import HandTracker, CupDetector
from pill_ingestion.ingestion_fsm import (
    IngestionSequenceDetector,
    IngestionConfig,
    _overlap_ratio,
    _distance_to_roi,
)

DEFAULT_VIDEO = PROJECT_ROOT / "dataset" / "taking_phill" / "IMG_0963.MOV"


def diagnose_hand_mouth(records: list[FrameRecord], overlap_th: float = 0.01, near_ratio: float = 3.0) -> dict:
    """매우 관대한 조건으로 hand_at_mouth 구간·hand_leave 구간 통계 (진단용)."""
    def at_mouth(roi: list[float], hand_bboxes: list[list[float]]) -> bool:
        if not roi or len(roi) < 4:
            return False
        for hb in hand_bboxes:
            if _overlap_ratio(roi, hb) >= overlap_th:
                return True
            if _distance_to_roi(roi, hb) <= near_ratio:
                return True
        return False

    valid = [r for r in records if r.roi_valid]
    at_mouth_flags = [at_mouth(r.mouth_roi, r.hand_bboxes) for r in valid]
    n_at = sum(at_mouth_flags)
    # 최대 연속 hand_at_mouth
    max_run_at = 0
    run = 0
    for v in at_mouth_flags:
        if v:
            run += 1
            max_run_at = max(max_run_at, run)
        else:
            run = 0
    # hand_at_mouth 직후 최대 연속 hand_leave
    max_run_leave = 0
    run = 0
    in_after_at = False
    for v in at_mouth_flags:
        if v:
            in_after_at = True
            run = 0
        else:
            if in_after_at:
                run += 1
                max_run_leave = max(max_run_leave, run)
    return {
        "total_frames": len(records),
        "valid_frames": len(valid),
        "frames_hand_at_mouth": n_at,
        "max_consecutive_hand_at_mouth": max_run_at,
        "max_consecutive_hand_leave_after": max_run_leave,
    }


@dataclass
class FrameRecord:
    time_sec: float
    mouth_roi: list[float]
    roi_valid: bool
    hand_bboxes: list[list[float]]
    cup_bbox: list[float]
    cup_conf: float


def collect_records(video_path: Path, max_frames: int = 9999) -> list[FrameRecord]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    records: list[FrameRecord] = []
    mouth_tracker = MouthROITracker(debug=False)
    hand_tracker = HandTracker()
    cup_detector = CupDetector()
    time_sec = 0.0
    n = 0
    try:
        while n < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            mouth_out: MouthROIResult = mouth_tracker.process(frame)
            hand_out = hand_tracker.update(frame)
            cup_out = cup_detector.update(frame)
            records.append(FrameRecord(
                time_sec=time_sec,
                mouth_roi=list(mouth_out["mouth_roi"]),
                roi_valid=mouth_out.get("roi_valid", True),
                hand_bboxes=list(hand_out["hand_bboxes"]),
                cup_bbox=list(cup_out.get("cup_bbox") or []),
                cup_conf=float(cup_out.get("cup_conf") or 0),
            ))
            time_sec += 1.0 / fps
            n += 1
    finally:
        mouth_tracker.close()
        hand_tracker.close()
    cap.release()
    return records


def run_fsm_on_records(records: list[FrameRecord], config: IngestionConfig) -> dict:
    """수집된 프레임 시퀀스에 대해 FSM 한 번 돌려서 최종 상태/결정 반환."""
    fsm = IngestionSequenceDetector(config=config)
    last_state = None
    for r in records:
        out = fsm.update(
            mouth_roi=r.mouth_roi,
            roi_valid=r.roi_valid,
            hand_bboxes=r.hand_bboxes,
            cup_bbox=r.cup_bbox,
            cup_conf=r.cup_conf,
            current_time_sec=r.time_sec,
        )
        last_state = out
        if out.get("final_decision") == "O":
            return {
                "success": True,
                "state": out["state"],
                "final_decision": out["final_decision"],
                "ingestion_time_sec": out.get("ingestion_time_sec"),
                "message": out.get("message"),
            }
    return {
        "success": False,
        "state": last_state.get("state") if last_state else "IDLE",
        "final_decision": last_state.get("final_decision") if last_state else None,
        "ingestion_time_sec": None,
        "message": last_state.get("message") if last_state else "",
    }


def main():
    video_path = DEFAULT_VIDEO
    if not video_path.exists():
        print(f"영상 없음: {video_path}", file=sys.stderr)
        return 1
    print("영상 1회 재생 중 (데이터 수집)...")
    records = collect_records(video_path)
    print(f"수집 프레임 수: {len(records)}, 구간 약 {records[-1].time_sec:.1f}초")

    diag = diagnose_hand_mouth(records)
    print(f"[진단] roi_valid 프레임: {diag['valid_frames']}, hand_at_mouth(관대): {diag['frames_hand_at_mouth']}프레임")
    print(f"       최대 연속 hand_at_mouth: {diag['max_consecutive_hand_at_mouth']}프레임, 최대 연속 hand_leave(이후): {diag['max_consecutive_hand_leave_after']}프레임")

    # roi_valid인 프레임만으로 FSM 시뮬레이션 (invalid 1프레임만 있어도 즉시 FAIL되므로)
    records_for_fsm = [r for r in records if r.roi_valid]
    print(f"튜닝: roi_valid 프레임만 사용 ({len(records_for_fsm)}프레임)")

    # hand-only 모드. 그리드 확대 (매우 관대한 값 포함)
    hand_overlap_thresholds = [0.01, 0.05, 0.08, 0.10, 0.15, 0.20]
    hand_near_ratios = [1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    hand_min_frames_list = [1, 2, 3, 4]
    hand_only_min_frames_list = [1, 2, 3, 4]  # 손이 입에 있던 최소 프레임 (속도 무관)
    hand_leave_min_frames_list = [1, 2, 3, 4]

    for hand_th in hand_overlap_thresholds:
        for hand_near in hand_near_ratios:
            for hand_min in hand_min_frames_list:
                for hand_only_frames in hand_only_min_frames_list:
                    for hand_leave in hand_leave_min_frames_list:
                        cfg = IngestionConfig(
                            cup_required=False,
                            hand_overlap_threshold=hand_th,
                            hand_near_mouth_max_ratio=hand_near,
                            hand_min_frames=hand_min,
                            hand_only_min_frames=hand_only_frames,
                            hand_leave_min_frames=hand_leave,
                        )
                        result = run_fsm_on_records(records_for_fsm, cfg)
                        if result["success"]:
                            print(f"\n[성공] O 도달")
                            print(f"  hand_overlap_threshold={hand_th}")
                            print(f"  hand_near_mouth_max_ratio={hand_near}")
                            print(f"  hand_min_frames={hand_min}")
                            print(f"  hand_only_min_frames={hand_only_frames}")
                            print(f"  hand_leave_min_frames={hand_leave}")
                            print(f"  ingestion_time_sec={result.get('ingestion_time_sec')}")
                            return apply_and_exit(
                                hand_th, hand_min, hand_only_frames, hand_leave, hand_near
                            )
    print("\n[실패] 그리드 내에서 O를 만드는 설정 없음. 그리드 확대 필요.")
    return 1


def apply_and_exit(
    hand_overlap_threshold: float,
    hand_min_frames: int,
    hand_only_min_frames: int,
    hand_leave_min_frames: int,
    hand_near_mouth_max_ratio: float = 1.4,
) -> int:
    """ingestion_fsm.py 기본값 수정 (정규식으로 현재 값 무관 치환)."""
    import re
    path = PROJECT_ROOT / "pill_ingestion" / "ingestion_fsm.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"hand_overlap_threshold: float = [\d.]+[^\n]*",
        f"hand_overlap_threshold: float = {hand_overlap_threshold}  # tune_ingestion 파인튜닝",
        text,
        count=1,
    )
    text = re.sub(
        r"hand_near_mouth_max_ratio: float = [\d.]+[^\n]*",
        f"hand_near_mouth_max_ratio: float = {hand_near_mouth_max_ratio}  # tune_ingestion 파인튜닝",
        text,
        count=1,
    )
    text = re.sub(
        r"hand_min_frames: int = \d+[^\n]*",
        f"hand_min_frames: int = {hand_min_frames}   # tune_ingestion 파인튜닝",
        text,
        count=1,
    )
    text = re.sub(
        r"hand_only_min_frames: int = \d+[^\n]*",
        f"hand_only_min_frames: int = {hand_only_min_frames}     # tune_ingestion 파인튜닝",
        text,
        count=1,
    )
    text = re.sub(
        r"hand_leave_min_frames: int = \d+[^\n]*",
        f"hand_leave_min_frames: int = {hand_leave_min_frames}    # tune_ingestion 파인튜닝",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print("\n기본값 반영 완료: pill_ingestion/ingestion_fsm.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

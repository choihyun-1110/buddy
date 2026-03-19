"""
손 검출 (추가 명세 2.2). MediaPipe Hand Landmarker Tasks API.
모델 없으면 빈 결과 반환 (로컬 전용, API 호출 없음).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
    from mediapipe.tasks.python.vision.core import image as image_lib
    _HAND_AVAILABLE = True
except (ImportError, AttributeError):
    _HAND_AVAILABLE = False

from mouth_roi_tracker.types import HandResult
from mouth_roi_tracker.model_utils import get_hand_landmarker_model_path


def _landmarks_to_bbox(landmarks: list, width: int, height: int, margin: float = 0.1) -> list[float]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    xmin = max(0, xmin - margin * w)
    ymin = max(0, ymin - margin * h)
    xmax = min(width, xmax + margin * w)
    ymax = min(height, ymax + margin * h)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


class HandTracker:
    def __init__(self, model_path: Optional[str] = None):
        self._landmarker = None
        self._frame_ts_ms = 0
        if not _HAND_AVAILABLE:
            return
        path = model_path or get_hand_landmarker_model_path()
        if path is None:
            return
        path = str(path)
        base_options = base_options_lib.BaseOptions(model_asset_path=path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def update(self, frame: np.ndarray) -> HandResult:
        """BGR 프레임 입력 → HandResult (랜드마크 포함)."""
        empty = HandResult(
            hand_bboxes=[], hand_confs=[], hand_count=0,
            hand_landmarks=[], handedness=[],
        )
        if self._landmarker is None:
            return empty
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._frame_ts_ms += 33
        hand_bboxes: list[list[float]] = []
        hand_confs: list[float] = []
        hand_landmarks: list[list[list[float]]] = []
        handedness: list[str] = []
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                bbox = _landmarks_to_bbox(landmarks, w, h)
                hand_bboxes.append(bbox)
                hand_confs.append(0.9)
                # 픽셀 좌표로 변환한 21개 랜드마크 [[x, y, z], ...]
                pts = [[lm.x * w, lm.y * h, lm.z] for lm in landmarks]
                hand_landmarks.append(pts)
                # handedness: result.handedness[i][0].display_name = "Left" or "Right"
                side = "Right"
                if result.handedness and i < len(result.handedness):
                    cats = result.handedness[i]
                    if cats:
                        side = cats[0].display_name
                handedness.append(side)
        return HandResult(
            hand_bboxes=hand_bboxes,
            hand_confs=hand_confs,
            hand_count=len(hand_bboxes),
            hand_landmarks=hand_landmarks,
            handedness=handedness,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

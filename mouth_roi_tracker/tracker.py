"""
Mouth ROI Tracking Module (명세 Section 1~7).
MediaPipe Tasks API (FaceLandmarker) + Fallback + EMA Smoothing + Occlusion Guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
    from mediapipe.tasks.python.vision.core import image as image_lib
    _TASKS_AVAILABLE = True
except (ImportError, AttributeError):
    mp = None
    _TASKS_AVAILABLE = False

from .model_utils import get_face_landmarker_model_path
from .types import MouthROIResult

# MediaPipe Face Landmarker 입 주변 outer lip 랜드마크 인덱스 (기존 468 메시와 호환)
OUTER_LIP_LANDMARK_INDICES = [
    0, 17, 37, 39, 40, 61, 78, 84, 87, 88, 91, 95,
    185, 267, 269, 270, 291, 308, 314, 317, 318, 321, 324, 375, 402, 405, 409,
]


@dataclass
class TrackerState:
    """명세 Section 7. ROI State Machine."""
    roi_valid: bool
    roi_mode: str  # "FACEMESH" | "FALLBACK"
    frames_since_valid: int   # 얼굴 미검출 시 카운트 (N까지 이전 ROI 유지)
    frames_since_landmark_ok: int  # 랜드마크 실패 연속 카운트 (K 초과 시 FALLBACK)
    previous_roi: Optional[list[float]] = None  # [x1, y1, x2, y2]


def _clamp_roi(roi: list[float], width: int, height: int) -> list[float]:
    """ROI를 이미지 범위 안으로 클램프."""
    x1, y1, x2, y2 = roi
    return [
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    ]


def _area(roi: list[float]) -> float:
    """ROI 면적 (테스트/유틸)."""
    return (roi[2] - roi[0]) * (roi[3] - roi[1])


def _landmarks_bbox(landmarks: list[Any], width: int, height: int) -> list[float]:
    """랜드마크 리스트 전체의 픽셀 bbox [x_min, y_min, x_max, y_max]."""
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    return [min(xs), min(ys), max(xs), max(ys)]


class MouthROITracker:
    """
    명세에 따른 Mouth ROI 추적기.
    FaceLandmarker (Tasks API) → Mouth ROI (Primary) / Fallback → EMA → Occlusion Guard
    """

    def __init__(
        self,
        # Margin (명세 Section 3)
        margin_x_ratio: float = 0.15,
        margin_y_ratio: float = 0.20,
        # Fallback 비율 (명세 Section 4)
        fallback_x_min: float = 0.2,
        fallback_x_max: float = 0.8,
        fallback_y_min: float = 0.6,
        fallback_y_max: float = 1.0,
        # EMA (명세 Section 5)
        ema_alpha: float = 0.5,
        # State machine (명세 Section 6, 7)
        max_frames_keep_roi: int = 10,
        max_frames_no_face: int = 5,
        landmark_confidence_threshold: float = 0.5,
        landmark_jump_ratio: float = 2.0,
        occlusion_area_ratio: float = 2.0,
        debug: bool = False,
        model_path: Optional[str] = None,
    ):
        if not _TASKS_AVAILABLE or mp is None:
            raise ImportError("mediapipe (Tasks API) is required. Install with: pip install mediapipe")

        self.margin_x_ratio = margin_x_ratio
        self.margin_y_ratio = margin_y_ratio
        self.fallback_x_min = fallback_x_min
        self.fallback_x_max = fallback_x_max
        self.fallback_y_min = fallback_y_min
        self.fallback_y_max = fallback_y_max
        self.ema_alpha = ema_alpha
        self.max_frames_keep_roi = max_frames_keep_roi
        self.max_frames_no_face = max_frames_no_face
        self.landmark_confidence_threshold = landmark_confidence_threshold
        self.landmark_jump_ratio = landmark_jump_ratio
        self.occlusion_area_ratio = occlusion_area_ratio
        self.debug = debug

        path = model_path or str(get_face_landmarker_model_path())
        base_options = base_options_lib.BaseOptions(model_asset_path=path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._state: Optional[TrackerState] = None

    def _get_face_bbox_from_landmarks(
        self,
        landmarks: list[Any],
        width: int,
        height: int,
    ) -> list[float]:
        """랜드마크 전체로 face bbox 계산."""
        return _landmarks_bbox(landmarks, width, height)

    def _get_mouth_roi_from_landmarks(
        self,
        landmarks: list[Any],
        width: int,
        height: int,
    ) -> Optional[tuple[list[float], float]]:
        """입 랜드마크 → mouth ROI. Returns (roi, confidence) or None."""
        xs, ys = [], []
        for i in OUTER_LIP_LANDMARK_INDICES:
            if i >= len(landmarks):
                continue
            lm = landmarks[i]
            xs.append(lm.x * width)
            ys.append(lm.y * height)
        if not xs or not ys:
            return None
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        mx = self.margin_x_ratio * (xmax - xmin)
        my = self.margin_y_ratio * (ymax - ymin)
        roi = [xmin - mx, ymin - my, xmax + mx, ymax + my]
        return (roi, 1.0)

    def _is_landmark_inside_face_bbox(
        self,
        landmarks: list[Any],
        face_bbox: list[float],
        width: int,
        height: int,
    ) -> bool:
        x1, y1, x2, y2 = face_bbox
        for i in OUTER_LIP_LANDMARK_INDICES:
            if i >= len(landmarks):
                continue
            lm = landmarks[i]
            px, py = lm.x * width, lm.y * height
            if px < x1 or px > x2 or py < y1 or py > y2:
                return False
        return True

    def _landmark_jump_too_large(
        self,
        landmarks: list[Any],
        prev_roi: list[float],
        width: int,
        height: int,
    ) -> bool:
        if not prev_roi:
            return False
        prev_cx = (prev_roi[0] + prev_roi[2]) / 2
        prev_cy = (prev_roi[1] + prev_roi[3]) / 2
        prev_w = prev_roi[2] - prev_roi[0]
        prev_h = prev_roi[3] - prev_roi[1]
        threshold_w = prev_w * self.landmark_jump_ratio
        threshold_h = prev_h * self.landmark_jump_ratio
        xs = [landmarks[i].x * width for i in OUTER_LIP_LANDMARK_INDICES if i < len(landmarks)]
        ys = [landmarks[i].y * height for i in OUTER_LIP_LANDMARK_INDICES if i < len(landmarks)]
        if not xs or not ys:
            return True
        cur_cx = (min(xs) + max(xs)) / 2
        cur_cy = (min(ys) + max(ys)) / 2
        return abs(cur_cx - prev_cx) > threshold_w or abs(cur_cy - prev_cy) > threshold_h

    def _fallback_roi_from_face(self, face_bbox: list[float]) -> list[float]:
        x_min, y_min, x_max, y_max = face_bbox
        fw = x_max - x_min
        fh = y_max - y_min
        return [
            x_min + self.fallback_x_min * fw,
            y_min + self.fallback_y_min * fh,
            x_min + self.fallback_x_max * fw,
            y_min + self.fallback_y_max * fh,
        ]

    def _ema_smooth(
        self,
        raw_roi: list[float],
        prev_roi: Optional[list[float]],
    ) -> list[float]:
        if prev_roi is None:
            return raw_roi
        a = self.ema_alpha
        return [
            a * raw_roi[0] + (1 - a) * prev_roi[0],
            a * raw_roi[1] + (1 - a) * prev_roi[1],
            a * raw_roi[2] + (1 - a) * prev_roi[2],
            a * raw_roi[3] + (1 - a) * prev_roi[3],
        ]

    def _occlusion_reject(self, raw_roi: list[float], prev_roi: Optional[list[float]]) -> bool:
        if prev_roi is None:
            return False
        a_prev = _area(prev_roi)
        a_cur = _area(raw_roi)
        if a_prev <= 0:
            return False
        r = a_cur / a_prev
        return r > self.occlusion_area_ratio or r < 1.0 / self.occlusion_area_ratio

    def process(
        self,
        frame: np.ndarray,
        previous_roi: Optional[list[float]] = None,
        previous_state: Optional[TrackerState] = None,
    ) -> MouthROIResult:
        h, w = frame.shape[:2]
        state = previous_state or self._state
        prev_roi = previous_roi if previous_roi is not None else (state.previous_roi if state else None)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
        result: FaceLandmarkerResult = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._frame_ts_ms += 33  # 약 30fps 가정

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            if state and state.frames_since_valid < self.max_frames_no_face and prev_roi:
                roi = self._ema_smooth(prev_roi, state.previous_roi)
                roi = _clamp_roi(roi, w, h)
                mode = state.roi_mode
                self._state = TrackerState(
                    roi_valid=False,
                    roi_mode=mode,
                    frames_since_valid=state.frames_since_valid + 1,
                    frames_since_landmark_ok=getattr(state, "frames_since_landmark_ok", 0),
                    previous_roi=roi,
                )
                return MouthROIResult(mouth_roi=roi, roi_mode=mode, confidence=0.0, roi_valid=False, lip_landmarks=[])
            # 화면 고정 ROI: 디버그/표시용만. 판정 로직에는 사용 안 함(roi_valid=False).
            fallback = [w * 0.35, h * 0.5, w * 0.65, h * 0.9]
            self._state = TrackerState(
                roi_valid=False,
                roi_mode="FALLBACK",
                frames_since_valid=0,
                frames_since_landmark_ok=0,
                previous_roi=fallback,
            )
            return MouthROIResult(mouth_roi=fallback, roi_mode="FALLBACK", confidence=0.0, roi_valid=False, lip_landmarks=[])

        landmarks = result.face_landmarks[0]
        face_bbox = self._get_face_bbox_from_landmarks(landmarks, w, h)
        # 삼킴 감지용 핵심 입술 landmark (픽셀 좌표)
        # 13=상단 내측, 14=하단 내측, 61=좌 입꼬리, 291=우 입꼬리
        _LIP_KEY = [13, 14, 61, 291]
        lip_lm: list[list[float]] = (
            [[landmarks[i].x * w, landmarks[i].y * h] for i in _LIP_KEY]
            if all(i < len(landmarks) for i in _LIP_KEY)
            else []
        )

        raw_roi: Optional[list[float]] = None
        roi_mode = "FALLBACK"
        confidence = 0.5
        landmark_ok = False

        if not self._is_landmark_inside_face_bbox(landmarks, face_bbox, w, h):
            pass
        elif self._landmark_jump_too_large(landmarks, prev_roi or [0, 0, w, h], w, h):
            pass
        else:
            primary = self._get_mouth_roi_from_landmarks(landmarks, w, h)
            if primary is not None:
                raw_roi, confidence = primary
                if not self._occlusion_reject(raw_roi, prev_roi):
                    roi_mode = "FACEMESH"
                    landmark_ok = True

        if not landmark_ok:
            prev_landmark_ok = getattr(state, "frames_since_landmark_ok", 0) if state else 0
            fail_count = prev_landmark_ok + 1
            if fail_count <= self.max_frames_keep_roi and prev_roi is not None:
                smoothed = self._ema_smooth(prev_roi, state.previous_roi if state else None)
                smoothed = _clamp_roi(smoothed, w, h)
                self._state = TrackerState(
                    roi_valid=True,
                    roi_mode=state.roi_mode if state else "FALLBACK",
                    frames_since_valid=0,
                    frames_since_landmark_ok=fail_count,
                    previous_roi=smoothed,
                )
                if self.debug:
                    self._draw_debug(frame, face_bbox, smoothed, self._state.roi_mode)
                return MouthROIResult(
                    mouth_roi=smoothed,
                    roi_mode=self._state.roi_mode,
                    confidence=0.5,
                    roi_valid=True,
                    lip_landmarks=lip_lm,
                )
            raw_roi = self._fallback_roi_from_face(face_bbox)
            confidence = 0.5
            frames_since_landmark_ok = 0
        else:
            frames_since_landmark_ok = 0

        if raw_roi is None:
            raw_roi = self._fallback_roi_from_face(face_bbox)
            confidence = 0.5

        smoothed = self._ema_smooth(raw_roi, prev_roi)
        smoothed = _clamp_roi(smoothed, w, h)
        self._state = TrackerState(
            roi_valid=True,
            roi_mode=roi_mode,
            frames_since_valid=0,
            frames_since_landmark_ok=frames_since_landmark_ok,
            previous_roi=smoothed,
        )
        if self.debug:
            self._draw_debug(frame, face_bbox, smoothed, roi_mode)
        return MouthROIResult(mouth_roi=smoothed, roi_mode=roi_mode, confidence=confidence, roi_valid=True, lip_landmarks=lip_lm)

    def _draw_debug(
        self,
        frame: np.ndarray,
        face_bbox: Optional[list[float]],
        mouth_roi: list[float],
        mode: str,
    ) -> None:
        vis = frame.copy()
        if face_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = [int(v) for v in mouth_roi]
        color = (0, 255, 255) if mode == "FACEMESH" else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, mode, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("MouthROITracker (debug)", vis)
        cv2.waitKey(1)

    def get_state(self) -> Optional[TrackerState]:
        return self._state

    def close(self) -> None:
        self._landmarker.close()
        if self.debug:
            cv2.destroyAllWindows()

    def __enter__(self) -> MouthROITracker:
        return self

    def __exit__(self, *args) -> None:
        self.close()

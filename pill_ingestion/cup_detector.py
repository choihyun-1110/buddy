"""
컵/병 검출 (추가 명세 2.3). YOLO (ultralytics) bottle/cup 클래스. API 없이 로컬에서만 추론.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from mouth_roi_tracker.types import CupResult

# COCO 0-based: 39=bottle, 40=wine_glass, 41=cup
CUP_CLASS_IDS = (39, 40, 41)

# 프로젝트 내 models 폴더에 두면 자동 사용 (네트워크 불필요). 없으면 ultralytics 캐시에서 로드 시 최초 1회 다운로드.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOCAL_MODEL = _PROJECT_ROOT / "models" / "yolov8n.pt"


class CupDetector:
    """YOLO 기반 컵/병 검출. model_path 미지정 시 models/yolov8n.pt → 없으면 'yolov8n.pt'(캐시/최초 다운로드)."""
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.05,
    ):
        self._model = None
        self._load_error: Optional[str] = None
        self.conf_threshold = conf_threshold
        if not _YOLO_AVAILABLE:
            self._load_error = "ultralytics 미설치. pip install ultralytics 후 다시 실행하세요."
            return
        path = model_path or _DEFAULT_LOCAL_MODEL
        if path == _DEFAULT_LOCAL_MODEL and not _DEFAULT_LOCAL_MODEL.exists():
            path = "yolov8n.pt"  # 캐시/최초 1회 다운로드 (네트워크 필요)
        try:
            self._model = YOLO(str(path))
        except Exception as e:
            self._model = None
            self._load_error = str(e)
            if path == "yolov8n.pt":
                self._load_error += " (최초 실행 시 모델 다운로드 실패일 수 있음. models/README.md 참고해 models/yolov8n.pt 를 받아 두세요.)"

    def is_loaded(self) -> bool:
        return self._model is not None

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        if len(bbox) < 4:
            return (0.0, 0.0)
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def update(self, frame: np.ndarray, mouth_roi: Optional[list[float]] = None) -> CupResult:
        """BGR 프레임 → 컵/병 bbox 하나. mouth_roi 있으면 입 근처 컵 우선 (손에 잡힌 컵이 입 쪽일 때)."""
        if self._model is None:
            return CupResult(cup_bbox=[], cup_conf=0.0)
        # 손에 잡힌 컵은 가려져서 conf 낮음 → YOLO conf 매우 낮게
        results = self._model.predict(
            frame,
            verbose=False,
            classes=list(CUP_CLASS_IDS),
            imgsz=1280,
            conf=0.01,
        )
        candidates: list[tuple[list[float], float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for i, cls in enumerate(r.boxes.cls):
                if int(cls) not in CUP_CLASS_IDS:
                    continue
                conf = float(r.boxes.conf[i])
                if conf < self.conf_threshold:
                    continue
                xyxy = r.boxes.xyxy[i].cpu().numpy()
                box = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                candidates.append((box, conf))
        if not candidates:
            return CupResult(cup_bbox=[], cup_conf=0.0)
        # 입 근처 컵 우선: mouth_roi 있으면 입 중심과 가까운 컵 선택 (손으로 컵 들고 입 쪽으로 가져올 때)
        if mouth_roi and len(mouth_roi) >= 4:
            mx = (mouth_roi[0] + mouth_roi[2]) / 2
            my = (mouth_roi[1] + mouth_roi[3]) / 2
            def dist(box: list[float]) -> float:
                cx, cy = self._center(box)
                return (cx - mx) ** 2 + (cy - my) ** 2
            best_box, best_conf = min(candidates, key=lambda x: dist(x[0]))
        else:
            best_box, best_conf = max(candidates, key=lambda x: x[1])
        return CupResult(cup_bbox=best_box, cup_conf=best_conf)

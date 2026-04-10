"""
컵/병 검출. YOLO PyTorch → ONNX Runtime 우선, 없으면 ultralytics 폴백.

최적화:
  1. ONNX Runtime (onnxruntime) 우선 사용 — PyTorch 오버헤드 제거
  2. imgsz 320 (기존 1280 대비 ~9배 빠름)
  3. 호출 측에서 N프레임 스킵 가능 (CupDetector는 매번 정직하게 추론)
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# ── ONNX Runtime (우선) ───────────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# ── ultralytics (폴백) ────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from mouth_roi_tracker.types import CupResult

# COCO 0-based: 39=bottle, 40=wine_glass, 41=cup
CUP_CLASS_IDS = {39, 40, 41}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ONNX  = _PROJECT_ROOT / "models" / "yolov8n.onnx"
_DEFAULT_PT    = _PROJECT_ROOT / "models" / "yolov8n.pt"
_IMGSZ = 320   # 기존 1280 → 320 (~9배 빠름)


# ── ONNX Runtime 추론 헬퍼 ────────────────────────────────────

def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """이미지를 비율 유지하며 size×size 로 리사이즈 (letterbox)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def _ort_detect(
    session: "ort.InferenceSession",
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> list[tuple[list[float], float]]:
    """ONNX Runtime으로 YOLOv8 추론 → [(bbox_xyxy, conf), ...]"""
    img, scale, (px, py) = _letterbox(frame_bgr, _IMGSZ)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(img_rgb.transpose(2, 0, 1), 0)  # [1,3,H,W]

    outputs = session.run(None, {session.get_inputs()[0].name: inp})
    preds = outputs[0]  # [1, 4+nc, 8400] or [1, 8400, 4+nc]

    if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(0, 2, 1)
    preds = preds[0]  # [8400, 4+nc]

    results = []
    h_orig, w_orig = frame_bgr.shape[:2]
    for pred in preds:
        cls_scores = pred[4:]
        cid = int(np.argmax(cls_scores))
        if cid not in CUP_CLASS_IDS:
            continue
        conf = float(cls_scores[cid])
        if conf < conf_threshold:
            continue
        cx, cy, bw, bh = pred[:4]
        # letterbox 역변환 → 원본 픽셀 좌표
        x1 = (cx - bw / 2 - px) / scale
        y1 = (cy - bh / 2 - py) / scale
        x2 = (cx + bw / 2 - px) / scale
        y2 = (cy + bh / 2 - py) / scale
        x1 = max(0.0, min(x1, w_orig))
        y1 = max(0.0, min(y1, h_orig))
        x2 = max(0.0, min(x2, w_orig))
        y2 = max(0.0, min(y2, h_orig))
        results.append(([x1, y1, x2, y2], conf))
    return results


class CupDetector:
    """
    ONNX Runtime 우선 → ultralytics 폴백.
    imgsz=320 고정으로 Pi4에서도 실용적인 속도 확보.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.05,
    ):
        self.conf_threshold = conf_threshold
        self._ort_session = None
        self._yolo_model = None
        self._load_error: Optional[str] = None
        self._backend: str = "none"

        # 1) ONNX Runtime 시도
        if _ORT_AVAILABLE:
            onnx_path = Path(model_path) if model_path else _DEFAULT_ONNX
            if onnx_path.exists():
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.inter_op_num_threads = 1
                    self._ort_session = ort.InferenceSession(
                        str(onnx_path),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self._backend = "onnxruntime"
                    print(f"[CupDetector] backend=ONNX Runtime  ({onnx_path.name})")
                    return
                except Exception as e:
                    self._load_error = f"ORT load failed: {e}"

        # 2) ultralytics 폴백
        if _YOLO_AVAILABLE:
            pt_path = Path(model_path) if model_path else _DEFAULT_PT
            if not pt_path.exists():
                pt_path = Path("yolov8n.pt")
            try:
                self._yolo_model = _YOLO(str(pt_path))
                self._backend = "ultralytics"
                print(f"[CupDetector] backend=ultralytics (imgsz={_IMGSZ})")
                return
            except Exception as e:
                self._load_error = str(e)

        if not self._load_error:
            self._load_error = "onnxruntime/ultralytics 둘 다 없음"

    def is_loaded(self) -> bool:
        return self._backend != "none"

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def update(self, frame: np.ndarray, mouth_roi: Optional[list[float]] = None) -> CupResult:
        if not self.is_loaded():
            return CupResult(cup_bbox=[], cup_conf=0.0)

        # 추론
        if self._backend == "onnxruntime":
            candidates = _ort_detect(self._ort_session, frame, self.conf_threshold)
        else:
            results = self._yolo_model.predict(
                frame, verbose=False,
                classes=list(CUP_CLASS_IDS),
                imgsz=_IMGSZ, conf=0.01,
            )
            candidates = []
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
                    candidates.append(([float(v) for v in xyxy], conf))

        if not candidates:
            return CupResult(cup_bbox=[], cup_conf=0.0)

        # 입 근처 컵 우선 선택
        if mouth_roi and len(mouth_roi) >= 4:
            mx, my = (mouth_roi[0] + mouth_roi[2]) / 2, (mouth_roi[1] + mouth_roi[3]) / 2
            best_box, best_conf = min(candidates, key=lambda x: (
                (self._center(x[0])[0] - mx) ** 2 + (self._center(x[0])[1] - my) ** 2
            ))
        else:
            best_box, best_conf = max(candidates, key=lambda x: x[1])

        return CupResult(cup_bbox=best_box, cup_conf=best_conf)

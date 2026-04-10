"""
알약 인식 모듈 (pill_recognition_ai 통합).

YOLOv8n ONNX 모델로 22종 알약을 검출한다.
라즈베리파이 배포 최적화: 416x416, CPU ONNX Runtime, 배치 1.

사용 시점:
    HAND_TO_MOUTH 상태에서 손이 열려 있을 때 (gripping=False)
    손 bbox ROI를 crop해 알약 감지.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "pill_recognition_ai"
    / "results"
    / "training_20251031_201041"
    / "best_model.onnx"
)

# 학습 시 사용한 22개 클래스 (data_config.yaml 기준)
PILL_CLASSES = [
    "29002", "34342", "37990", "39916", "40122",
    "40720", "40767", "40792", "40837", "40949",
    "40953", "40990", "40991", "41097", "41107",
    "41169", "41170", "41172", "41207", "41225",
    "41327", "41344",
]


@dataclass
class PillDetection:
    class_id: int
    class_name: str   # 품목코드 (e.g. "40122")
    confidence: float
    bbox: list[int]   # [x1, y1, x2, y2] 원본 프레임 픽셀 좌표


@dataclass
class PillResult:
    detections: list[PillDetection] = field(default_factory=list)
    inference_ms: float = 0.0
    model_loaded: bool = False
    error: str = ""

    @property
    def best(self) -> Optional[PillDetection]:
        """신뢰도 최고 감지 결과."""
        return max(self.detections, key=lambda d: d.confidence) if self.detections else None


class PillRecognizer:
    """
    YOLOv8n ONNX 기반 알약 인식기.

    Parameters
    ----------
    model_path:
        .onnx 파일 경로. None이면 기본 경로(pill_recognition_ai/results/...) 사용.
    input_size:
        추론 입력 크기. 라즈베리파이는 416, 노트북은 640 권장.
    conf_threshold:
        신뢰도 임계값.
    nms_threshold:
        NMS IoU 임계값.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        input_size: int = 416,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ):
        self._input_size = input_size
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._session = None
        self._input_name = ""
        self._load_error = ""

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._load_model(path)

    def _load_model(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self._load_error = "onnxruntime 미설치. pip install onnxruntime"
            return

        if not path.exists():
            self._load_error = f"모델 파일 없음: {path}"
            return

        try:
            # CPU만 사용 (라즈베리파이 호환)
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            # 모델에 고정 입력 크기가 있으면 그걸 우선 사용
            shape = self._session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                self._input_size = shape[2]
            print(f"[PillRecognizer] 로드 완료: {path.name}, 입력 {self._input_size}×{self._input_size}")
        except Exception as e:
            self._load_error = str(e)
            print(f"[PillRecognizer] 로드 실패: {e}")

    def is_loaded(self) -> bool:
        return self._session is not None

    def predict(
        self,
        frame: np.ndarray,
        roi: Optional[list[float]] = None,
    ) -> PillResult:
        """
        Parameters
        ----------
        frame:
            BGR 이미지 (OpenCV).
        roi:
            [x1, y1, x2, y2] 픽셀 좌표. 지정 시 해당 영역만 crop 후 추론.
            None이면 전체 프레임 사용.
        """
        if not self.is_loaded():
            return PillResult(model_loaded=False, error=self._load_error)

        orig_h, orig_w = frame.shape[:2]
        offset_x, offset_y = 0, 0

        if roi and len(roi) == 4:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

        h, w = frame.shape[:2]
        inp = self._preprocess(frame)

        t0 = time.time()
        try:
            outputs = self._session.run(None, {self._input_name: inp})
        except Exception as e:
            return PillResult(model_loaded=True, error=str(e))
        ms = (time.time() - t0) * 1000

        detections = self._postprocess(outputs, w, h, offset_x, offset_y)
        return PillResult(detections=detections, inference_ms=round(ms, 1), model_loaded=True)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        s = self._input_size
        resized = cv2.resize(img, (s, s))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        return np.expand_dims(norm.transpose(2, 0, 1), axis=0)

    def _postprocess(
        self,
        outputs: list,
        orig_w: int,
        orig_h: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[PillDetection]:
        preds = outputs[0]
        # YOLOv8 출력 형태: [1, 4+nc, 8400] or [1, 8400, 4+nc]
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(0, 2, 1)
        preds = preds[0]  # [8400, 4+nc]

        nc = len(PILL_CLASSES)
        boxes, scores, class_ids = [], [], []

        for pred in preds:
            cx, cy, bw, bh = pred[:4]
            class_scores = pred[4:4 + nc]
            cid = int(np.argmax(class_scores))
            conf = float(class_scores[cid])
            if conf < self._conf:
                continue
            x1 = int((cx - bw / 2) * orig_w) + offset_x
            y1 = int((cy - bh / 2) * orig_h) + offset_y
            x2 = int((cx + bw / 2) * orig_w) + offset_x
            y2 = int((cy + bh / 2) * orig_h) + offset_y
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] for NMS
            scores.append(conf)
            class_ids.append(cid)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self._conf, self._nms)
        if len(indices) == 0:
            return []

        result = []
        for i in indices.flatten():
            x, y, bw2, bh2 = boxes[i]
            result.append(PillDetection(
                class_id=class_ids[i],
                class_name=PILL_CLASSES[class_ids[i]],
                confidence=round(scores[i], 3),
                bbox=[x, y, x + bw2, y + bh2],
            ))
        return result

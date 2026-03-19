"""타입 및 출력 스키마 정의 (명세 8. Interface Specification + 추가 명세서)."""
from typing import Literal, TypedDict


class MouthROIResult(TypedDict):
    """트래커 출력 (명세 Section 8). 추가 명세: roi_valid (이벤트 감지 허용 여부)."""
    mouth_roi: list[float]  # [x1, y1, x2, y2]
    roi_mode: Literal["FACEMESH", "FALLBACK"]
    confidence: float
    roi_valid: bool  # True일 때만 hand/cup 이벤트 판정 사용 (고정 ROI 사용 시 False)
    lip_landmarks: list  # [[upper_inner_x,y], [lower_inner_x,y], [left_corner_x,y], [right_corner_x,y]] (px)


class HandResult(TypedDict):
    """손 검출 결과 (추가 명세 2.2)."""
    hand_bboxes: list[list[float]]  # 각 손 [x1,y1,x2,y2]
    hand_confs: list[float]
    hand_count: int
    hand_landmarks: list[list[list[float]]]  # 각 손의 21개 랜드마크 [[x,y,z], ...] (픽셀 좌표)
    handedness: list[str]  # "Left" | "Right"


class CupResult(TypedDict):
    """컵 검출 결과 (추가 명세 2.3)."""
    cup_bbox: list[float]  # [x1,y1,x2,y2] 또는 []
    cup_conf: float


class IngestionState(TypedDict):
    """복용 시퀀스 FSM 상태 (추가 명세)."""
    state: str  # IDLE | HAND_TO_MOUTH | CUP_TO_MOUTH | DRINKING | INGESTION_COMPLETE | FAIL
    event_triggered: bool
    final_decision: Literal["O", "X", "UNKNOWN"] | None  # O=복용완료, X=미복용, UNKNOWN=판정불가
    message: str
    ingestion_time_sec: float | None  # 복용 완료 시 소요 시간

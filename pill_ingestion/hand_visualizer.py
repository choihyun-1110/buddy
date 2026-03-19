"""
손 랜드마크 시각화 + 그리핑 감지.

MediaPipe 21개 keypoint를 AR 스타일 뼈대로 그리고,
손가락 굽힘 각도로 컵/물체 그리핑 여부를 판별한다.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 랜드마크 인덱스 (MediaPipe Hand Landmarker)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 뼈대 연결 (segment → 색상 이름)
_CONNECTIONS: list[tuple[int, int]] = [
    # 엄지
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # 검지
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # 중지
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # 약지
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # 새끼
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # 손바닥 가로 연결
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]

# 손가락별 뼈대 색상 (BGR)
_FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    THUMB_CMC: (0, 140, 255),    # 엄지: 주황
    THUMB_MCP: (0, 140, 255),
    THUMB_IP:  (0, 140, 255),
    THUMB_TIP: (0, 140, 255),
    INDEX_MCP: (255, 230, 0),    # 검지: 하늘
    INDEX_PIP: (255, 230, 0),
    INDEX_DIP: (255, 230, 0),
    INDEX_TIP: (255, 230, 0),
    MIDDLE_MCP: (0, 255, 100),   # 중지: 연두
    MIDDLE_PIP: (0, 255, 100),
    MIDDLE_DIP: (0, 255, 100),
    MIDDLE_TIP: (0, 255, 100),
    RING_MCP:  (200, 0, 255),    # 약지: 보라
    RING_PIP:  (200, 0, 255),
    RING_DIP:  (200, 0, 255),
    RING_TIP:  (200, 0, 255),
    PINKY_MCP: (0, 80, 255),     # 새끼: 빨강
    PINKY_PIP: (0, 80, 255),
    PINKY_DIP: (0, 80, 255),
    PINKY_TIP: (0, 80, 255),
}
_PALM_COLOR = (200, 200, 200)    # 손바닥 연결: 회색
_JOINT_COLOR = (255, 255, 255)   # 관절 점: 흰색
_GRIP_JOINT_COLOR = (0, 255, 255)  # 굽힌 관절: 노랑


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------

def _pt(lm: Sequence[float]) -> tuple[int, int]:
    """랜드마크 [x, y, z] → 정수 픽셀 좌표."""
    return int(lm[0]), int(lm[1])


def _angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """세 점 a-b-c 에서 b를 꼭짓점으로 하는 각도(도)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_val))


# ---------------------------------------------------------------------------
# 그리핑 감지
# ---------------------------------------------------------------------------

# 각 손가락: (MCP, PIP, DIP, TIP)
_FINGER_JOINTS = [
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]


def detect_grip(landmarks: list[list[float]], curl_angle_threshold: float = 150.0) -> tuple[bool, list[bool]]:
    """
    손이 그리핑 상태인지 판별.

    각 손가락의 PIP 관절 각도를 계산해 `curl_angle_threshold`(기본 150도) 미만이면 굽힘으로 본다.
    4개 손가락 중 3개 이상 굽히면 gripping=True.

    Returns:
        (gripping: bool, finger_curled: list[bool])  # finger_curled 순서: [검지,중지,약지,새끼]
    """
    if len(landmarks) < 21:
        return False, [False, False, False, False]

    finger_curled: list[bool] = []
    for mcp, pip, dip, tip in _FINGER_JOINTS:
        # PIP 각도: mcp-pip-dip
        angle = _angle_deg(landmarks[mcp], landmarks[pip], landmarks[dip])
        finger_curled.append(angle < curl_angle_threshold)

    gripping = sum(finger_curled) >= 3
    return gripping, finger_curled


def detect_pinch(landmarks: list[list[float]], threshold_px: float = 40.0) -> bool:
    """엄지 TIP ↔ 검지 TIP 거리가 threshold_px 미만이면 핀치."""
    if len(landmarks) < 21:
        return False
    tx, ty = landmarks[THUMB_TIP][0], landmarks[THUMB_TIP][1]
    ix, iy = landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]
    dist = math.hypot(tx - ix, ty - iy)
    return dist < threshold_px


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def draw_hand_landmarks(
    frame: np.ndarray,
    landmarks: list[list[float]],
    handedness: str = "Right",
    gripping: bool = False,
    finger_curled: list[bool] | None = None,
    alpha: float = 0.85,
) -> None:
    """
    AR 스타일 손 뼈대를 frame 위에 그린다 (in-place).

    - 뼈대 선: 손가락별 고유 색상
    - 관절 점: 흰색 원, 굽힌 관절은 노란색
    - 그리핑 시: 손목 위에 "GRIP" 텍스트 + 반투명 파란 오버레이
    """
    if len(landmarks) < 21:
        return

    if finger_curled is None:
        _, finger_curled = detect_grip(landmarks)

    overlay = frame.copy()

    # 1) 뼈대 선 그리기
    palm_segs = {(INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)}
    for a_idx, b_idx in _CONNECTIONS:
        pt_a = _pt(landmarks[a_idx])
        pt_b = _pt(landmarks[b_idx])
        if (a_idx, b_idx) in palm_segs or (b_idx, a_idx) in palm_segs:
            color = _PALM_COLOR
        else:
            color = _FINGER_COLORS.get(b_idx, _PALM_COLOR)
        cv2.line(overlay, pt_a, pt_b, color, 2, cv2.LINE_AA)

    # 2) 관절 점
    # 굽힌 손가락 PIP 인덱스 목록
    curled_pips = set()
    for i, (mcp, pip, dip, tip) in enumerate(_FINGER_JOINTS):
        if finger_curled and i < len(finger_curled) and finger_curled[i]:
            curled_pips.add(pip)

    for idx, lm in enumerate(landmarks):
        pt = _pt(lm)
        color = _GRIP_JOINT_COLOR if idx in curled_pips else _JOINT_COLOR
        radius = 5 if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 4
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, radius, (30, 30, 30), 1, cv2.LINE_AA)  # 테두리

    # 3) 그리핑 표시
    if gripping:
        wrist_pt = _pt(landmarks[WRIST])
        label_pos = (wrist_pt[0] - 30, wrist_pt[1] + 20)
        cv2.putText(overlay, "GRIP", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # 4) 손 라벨 (L/R)
    wrist_pt = _pt(landmarks[WRIST])
    label = handedness[0]  # "L" or "R"
    label_color = (100, 220, 255) if handedness == "Right" else (255, 180, 100)
    cv2.putText(overlay, label, (wrist_pt[0] - 8, wrist_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

    # 반투명 합성
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_all_hands(
    frame: np.ndarray,
    hand_landmarks_list: list[list[list[float]]],
    handedness_list: list[str],
    cup_held_flags: list[bool] | None = None,
) -> list[bool]:
    """
    감지된 모든 손을 그리고 그리핑 상태 리스트를 반환.

    Returns:
        gripping_list: 각 손의 gripping bool
    """
    gripping_list: list[bool] = []
    for i, (landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
        gripping, finger_curled = detect_grip(landmarks)
        # 컵을 쥐고 있을 때 wrist 위에 CUP 표시를 위해 label 오버라이드
        draw_hand_landmarks(frame, landmarks, handedness, gripping, finger_curled)
        if cup_held_flags and i < len(cup_held_flags) and cup_held_flags[i]:
            wrist_pt = _pt(landmarks[WRIST])
            cv2.putText(frame, "CUP", (wrist_pt[0] - 25, wrist_pt[1] + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        gripping_list.append(gripping)
    return gripping_list


# ---------------------------------------------------------------------------
# 컵 쥐기 판별 유틸
# ---------------------------------------------------------------------------

def bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    """두 bbox [x1,y1,x2,y2] 교집합 / min(area_a, area_b)."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def is_holding_cup(
    hand_bbox: list[float],
    cup_bbox: list[float],
    gripping: bool,
    overlap_threshold: float = 0.25,
) -> bool:
    """손이 그리핑 상태이고 컵 bbox와 충분히 겹치면 True."""
    if not gripping or not cup_bbox or len(cup_bbox) < 4:
        return False
    return bbox_overlap_ratio(hand_bbox, cup_bbox) >= overlap_threshold

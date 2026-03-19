"""Face/Hand Landmarker 모델 경로. API/네트워크 호출 없이 로컬 파일만 사용."""
from pathlib import Path
from typing import Optional

# 패키지 기준 프로젝트 루트
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"

# 수동 다운로드 시 사용할 URL (코드에서 자동 호출하지 않음)
FACE_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
HAND_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def get_face_landmarker_model_path() -> Path:
    """face_landmarker.task 로컬 경로 반환. 없으면 에러 (다운로드 안 함)."""
    if not FACE_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {FACE_LANDMARKER_MODEL}\n"
            f"로컬에서만 실행하려면 한 번 수동으로 받아서 둬야 합니다.\n"
            f"  mkdir -p {MODELS_DIR}\n"
            f"  curl -L -o {FACE_LANDMARKER_MODEL} \"{FACE_LANDMARKER_DOWNLOAD_URL}\"\n"
            f"또는 브라우저에서 위 URL 로 받은 파일을 {FACE_LANDMARKER_MODEL} 에 저장하세요."
        )
    return FACE_LANDMARKER_MODEL


def get_hand_landmarker_model_path() -> Optional[Path]:
    """hand_landmarker.task 로컬 경로. 없으면 None (손 검출 비활성)."""
    if HAND_LANDMARKER_MODEL.exists():
        return HAND_LANDMARKER_MODEL
    return None

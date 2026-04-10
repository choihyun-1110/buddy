"""
YOLOv8 PyTorch → ONNX 변환 스크립트.

실행:
    python scripts/export_yolo_onnx.py

결과:
    models/yolov8n.onnx  (CupDetector가 자동으로 우선 사용)
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
pt_path   = PROJECT_ROOT / "models" / "yolov8n.pt"
onnx_path = PROJECT_ROOT / "models" / "yolov8n.onnx"

if not pt_path.exists():
    print(f"Not found: {pt_path}")
    print("Run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" to download.")
    raise SystemExit(1)

model = YOLO(str(pt_path))
model.export(format="onnx", imgsz=320, opset=12, simplify=True)

exported = pt_path.with_suffix(".onnx")
if exported.exists() and not onnx_path.exists():
    exported.rename(onnx_path)

print(f"Saved: {onnx_path}")

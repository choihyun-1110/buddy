"""
라즈베리파이 모드로 특정 이미지 추론 및 시각화
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from typing import List, Tuple

# ONNX 모델 경로
ONNX_MODEL = "results/training_20251031_201041/best_model.onnx"
IMAGE_PATH = "dataset_augmented_v3/test/images/29002_0004_aug_0001.png"
OUTPUT_DIR = Path("results/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# 클래스 이름
CLASS_NAMES = [
    "29002", "34342", "37990", "39916", "40122", "40720", "40767", "40792",
    "40837", "40949", "40953", "40990", "40991", "41097", "41107", "41169",
    "41170", "41172", "41207", "41225", "41327", "41344"
]

def load_onnx_model(onnx_path: str):
    """ONNX 모델 로드 (CPU 전용)"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 라즈베리파이 모드: CPU만 사용
    providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers
    )
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = (input_shape[2], input_shape[3])  # (width, height)
    
    print(f"✅ ONNX 모델 로드 완료 (CPU 모드)")
    print(f"   입력 크기: {input_size}")
    
    return session, input_name, input_size

def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """이미지 전처리"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    original_shape = img.shape[:2]
    
    # 리사이즈
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 정규화 (0-255 -> 0-1)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # (H, W, C) -> (C, H, W)
    img_transposed = img_normalized.transpose(2, 0, 1)
    
    # 배치 차원 추가 (1, C, H, W)
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, original_shape

def postprocess_yolo_output(outputs: list, conf_threshold: float = 0.25, num_classes: int = 22) -> list:
    """YOLO 출력 후처리 (YOLOv8 형식)"""
    detections = []
    
    if len(outputs) > 0:
        output = outputs[0]  # [batch, num_features, num_anchors]
        
        if output.ndim == 3:
            output = output[0]  # 배치 차원 제거: [num_features, num_anchors]
        
        # YOLOv8 출력 형식: [batch, num_features, num_anchors]
        # num_features = 4 (bbox) + num_classes
        # 출력을 transpose: [num_anchors, num_features]
        if output.ndim == 2:
            output = output.transpose(1, 0)  # [num_anchors, num_features]
        
        # 박스 좌표와 클래스 확률 분리
        boxes = output[:, :4]  # [num_anchors, 4] - 정규화된 xywh
        scores = output[:, 4:]  # [num_anchors, num_classes] - 클래스 확률
        
        # 각 앵커에 대해 최대 클래스 확률과 클래스 ID 찾기
        max_scores = np.max(scores, axis=1)  # [num_anchors]
        class_ids = np.argmax(scores, axis=1)  # [num_anchors]
        
        # 신뢰도 임계값 필터링
        valid_indices = max_scores > conf_threshold
        
        if np.any(valid_indices):
            valid_boxes = boxes[valid_indices]
            valid_scores = max_scores[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            for box, score, class_id in zip(valid_boxes, valid_scores, valid_class_ids):
                x_center, y_center, width, height = box
                
                detections.append({
                    'bbox': [float(x_center), float(y_center), float(width), float(height)],
                    'confidence': float(score),
                    'class_id': int(class_id)
                })
    
    return detections

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return np.array([])
    
    # xywh를 xyxy로 변환
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    boxes_xyxy = np.column_stack([x1, y1, x2, y2])
    
    # 면적 계산
    areas = (x2 - x1) * (y2 - y1)
    
    # 점수 순으로 정렬된 인덱스
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # IoU 계산
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[order[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[order[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[order[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # IoU 임계값 이하인 박스만 유지
        order = order[1:][iou <= iou_threshold]
    
    return np.array(keep)

def postprocess_yolo_output_with_nms(outputs: list, conf_threshold: float = 0.25, 
                                     iou_threshold: float = 0.45, num_classes: int = 22) -> list:
    """YOLO 출력 후처리 (NMS 포함)"""
    detections = []
    
    if len(outputs) > 0:
        output = outputs[0]  # [batch, num_features, num_anchors]
        
        if output.ndim == 3:
            output = output[0]  # 배치 차원 제거: [num_features, num_anchors]
        
        # YOLOv8 출력 형식: [batch, num_features, num_anchors]
        # num_features = 4 (bbox) + num_classes
        # 출력을 transpose: [num_anchors, num_features]
        if output.ndim == 2:
            output = output.transpose(1, 0)  # [num_anchors, num_features]
        
        # 박스 좌표와 클래스 확률 분리
        boxes = output[:, :4]  # [num_anchors, 4] - 정규화된 xywh
        scores = output[:, 4:]  # [num_anchors, num_classes] - 클래스 확률
        
        # 각 앵커에 대해 최대 클래스 확률과 클래스 ID 찾기
        max_scores = np.max(scores, axis=1)  # [num_anchors]
        class_ids = np.argmax(scores, axis=1)  # [num_anchors]
        
        # 신뢰도 임계값 필터링
        valid_indices = max_scores > conf_threshold
        
        if np.any(valid_indices):
            valid_boxes = boxes[valid_indices]
            valid_scores = max_scores[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            # NMS 적용
            nms_indices = nms(valid_boxes, valid_scores, iou_threshold)
            
            for idx in nms_indices:
                box = valid_boxes[idx]
                score = valid_scores[idx]
                class_id = valid_class_ids[idx]
                
                x_center, y_center, width, height = box
                
                detections.append({
                    'bbox': [float(x_center), float(y_center), float(width), float(height)],
                    'confidence': float(score),
                    'class_id': int(class_id)
                })
    
    return detections

def draw_detections(image_path: str, detections: list, output_path: str, original_shape: tuple, input_size: tuple):
    """감지 결과 시각화"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    orig_h, orig_w = original_shape
    input_w, input_h = input_size  # input_size는 (width, height) 형식
    
    # 스케일 비율 계산
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    
    # 이미지 크기 정보 (디버깅용, 필요시 주석 해제)
    # print(f"\n📐 이미지 크기 정보:")
    # print(f"   원본 크기: {orig_w}x{orig_h}")
    # print(f"   입력 크기: {input_w}x{input_h}")
    # print(f"   스케일 비율: {scale_x:.3f} x {scale_y:.3f}")
    
    for det in detections:
        x_center, y_center, width, height = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        # YOLO 출력 좌표 형식 확인 및 변환
        # 만약 이미 픽셀 좌표라면 (값이 1보다 크면)
        if x_center > 1.0 or y_center > 1.0 or width > 1.0 or height > 1.0:
            # 이미 픽셀 좌표 (640x640 기준)
            x_center_px = x_center
            y_center_px = y_center
            width_px = width
            height_px = height
        else:
            # 정규화된 좌표 (0-1)
            x_center_px = x_center * input_w
            y_center_px = y_center * input_h
            width_px = width * input_w
            height_px = height * input_h
        
        # 원본 이미지 크기로 스케일링 (이미지가 640x640이면 스케일링 불필요)
        if orig_w == input_w and orig_h == input_h:
            # 이미지 크기가 같으면 스케일링 불필요
            x_center_scaled = x_center_px
            y_center_scaled = y_center_px
            width_scaled = width_px
            height_scaled = height_px
        else:
            # 다른 크기면 스케일링 필요
            x_center_scaled = x_center_px * scale_x
            y_center_scaled = y_center_px * scale_y
            width_scaled = width_px * scale_x
            height_scaled = height_px * scale_y
        
        # xywh를 xyxy로 변환
        x1 = int(x_center_scaled - width_scaled / 2)
        y1 = int(y_center_scaled - height_scaled / 2)
        x2 = int(x_center_scaled + width_scaled / 2)
        y2 = int(y_center_scaled + height_scaled / 2)
        
        # 좌표 범위 제한
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))
        
        # 좌표 유효성 확인
        if x2 <= x1 or y2 <= y1:
            print(f"   ⚠️ 잘못된 박스 좌표: ({x1}, {y1}) -> ({x2}, {y2}), 건너뜀")
            continue
        
        # 바운딩 박스 그리기 (선 두께를 더 두껍게)
        color = (0, 255, 0)  # 초록색 (BGR)
        thickness = 3  # 선 두께
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 그리기
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        label = f"{class_name}: {conf:.2f}"
        
        # 텍스트 배경
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            img, 
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # 텍스트
        cv2.putText(
            img, 
            label, 
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )
    
    cv2.imwrite(str(output_path), img)
    print(f"✅ 결과 저장: {output_path}")

def main():
    print("="*60)
    print("🍓 라즈베리파이 모드 추론 및 시각화")
    print("="*60)
    
    # 모델 로드
    session, input_name, input_size = load_onnx_model(ONNX_MODEL)
    
    # 이미지 전처리
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    print(f"\n📸 이미지 로드: {image_path.name}")
    img_batch, original_shape = preprocess_image(str(image_path), input_size)
    
    # 추론 실행
    print("🔍 추론 실행 중 (CPU 모드)...")
    start_time = time.time()
    
    outputs = session.run(None, {input_name: img_batch})
    
    inference_time = time.time() - start_time
    
    print(f"   추론 시간: {inference_time*1000:.1f}ms")
    print(f"   예상 FPS: {1/inference_time:.1f}")
    
    # 후처리 (NMS 포함)
    detections = postprocess_yolo_output_with_nms(outputs, conf_threshold=0.25, iou_threshold=0.45)
    
    print(f"   감지된 객체: {len(detections)}개")
    for i, det in enumerate(detections, 1):
        class_name = CLASS_NAMES[det['class_id']] if det['class_id'] < len(CLASS_NAMES) else f"class_{det['class_id']}"
        print(f"   {i}. {class_name}: 신뢰도 {det['confidence']:.2f}")
    
    # 시각화
    output_path = OUTPUT_DIR / f"raspberry_pi_{image_path.stem}_prediction.png"
    draw_detections(str(image_path), detections, str(output_path), original_shape, input_size)
    
    print("\n✅ 완료!")
    print(f"   결과 이미지: {output_path}")

if __name__ == "__main__":
    main()


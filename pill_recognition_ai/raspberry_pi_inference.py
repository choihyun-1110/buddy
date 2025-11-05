"""
라즈베리파이에서 실행할 추론 스크립트
"""
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json


def load_onnx_model(model_path: str):
    """ONNX 모델 로드"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # CPU만 사용 (라즈베리파이는 GPU 없음)
    providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = (input_shape[2], input_shape[3])  # (width, height)
    
    print(f"✅ 모델 로드 완료")
    print(f"   입력 크기: {input_size}")
    
    return session, input_name, input_size


def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """이미지 전처리"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
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
    
    return img_batch


def postprocess_output(outputs: list, conf_threshold: float = 0.25) -> list:
    """YOLO 출력 후처리"""
    detections = []
    
    if len(outputs) > 0:
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]
        
        # 신뢰도 임계값 필터링
        confidences = output[:, 4]
        valid_indices = confidences > conf_threshold
        
        if np.any(valid_indices):
            valid_outputs = output[valid_indices]
            
            for det in valid_outputs:
                x_center, y_center, width, height, conf, class_id = det[:6]
                
                detections.append({
                    'bbox': [float(x_center), float(y_center), float(width), float(height)],
                    'confidence': float(conf),
                    'class_id': int(class_id)
                })
    
    return detections


def draw_results(image_path: str, detections: list, output_path: str = None):
    """결과 시각화"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    # 클래스 이름 (실제 클래스에 맞게 수정)
    class_names = [
        "29002", "34342", "37990", "39916", "40122", "40720", "40767", "40792",
        "40837", "40949", "40953", "40990", "40991", "41097", "41107", "41169",
        "41170", "41172", "41207", "41225", "41327", "41344"
    ]
    
    h, w = img.shape[:2]
    
    for det in detections:
        x_center, y_center, width, height = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 라벨 그리기
        label = f"{class_names[class_id] if class_id < len(class_names) else 'unknown'}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"✅ 결과 저장: {output_path}")
    else:
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="라즈베리파이 ONNX 추론")
    parser.add_argument("--model", required=True, help="ONNX 모델 경로")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--output", help="출력 이미지 경로 (선택)")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--json", help="결과를 JSON 파일로 저장 (선택)")
    
    args = parser.parse_args()
    
    # 모델 로드
    session, input_name, input_size = load_onnx_model(args.model)
    
    # 이미지 전처리
    print(f"📸 이미지 로드: {args.image}")
    img_batch = preprocess_image(args.image, input_size)
    
    # 추론
    print("🔍 추론 실행 중...")
    start_time = time.time()
    
    outputs = session.run(None, {input_name: img_batch})
    
    inference_time = time.time() - start_time
    
    # 후처리
    detections = postprocess_output(outputs, args.conf)
    
    print(f"✅ 추론 완료")
    print(f"   추론 시간: {inference_time*1000:.1f}ms")
    print(f"   감지된 객체: {len(detections)}개")
    
    for i, det in enumerate(detections, 1):
        print(f"   {i}. 클래스 {det['class_id']}: 신뢰도 {det['confidence']:.2f}")
    
    # 결과 저장
    if args.output:
        draw_results(args.image, detections, args.output)
    
    if args.json:
        result = {
            'image': args.image,
            'inference_time_ms': inference_time * 1000,
            'num_detections': len(detections),
            'detections': detections
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON 결과 저장: {args.json}")


if __name__ == "__main__":
    main()


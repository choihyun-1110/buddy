"""
라즈베리파이 배포 전 테스트 스크립트
PC에서 CPU 모드로 라즈베리파이 환경을 시뮬레이션하여 테스트합니다.
"""
import time
import numpy as np
import cv2
from pathlib import Path
import onnxruntime as ort
from typing import List, Tuple, Dict
import psutil


class RaspberryPiSimulator:
    """라즈베리파이 환경 시뮬레이터"""
    
    def __init__(self, onnx_model_path: str):
        """초기화"""
        self.onnx_model_path = Path(onnx_model_path)
        
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_model_path}")
        
        # ONNX Runtime 세션 생성 (CPU 전용)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 라즈베리파이는 GPU가 없으므로 CPU만 사용
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # 입력/출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 입력 크기 확인
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])  # (width, height)
        
        print(f"✅ ONNX 모델 로드 완료: {self.onnx_model_path.name}")
        print(f"   입력 크기: {self.input_size}")
        print(f"   실행 공급자: {providers}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """이미지 전처리"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 원본 크기 저장
        original_shape = img.shape[:2]
        
        # 리사이즈
        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        img_transposed = img_normalized.transpose(2, 0, 1)
        
        # 배치 차원 추가 (1, C, H, W)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, original_shape
    
    def postprocess_output(self, outputs: List[np.ndarray], conf_threshold: float = 0.25) -> List[Dict]:
        """YOLO 출력 후처리"""
        # YOLO ONNX 출력은 일반적으로 [batch, num_detections, 6] 형태
        # [x_center, y_center, width, height, confidence, class_id]
        
        detections = []
        
        if len(outputs) > 0:
            output = outputs[0]  # 첫 번째 출력
            if output.ndim == 3:
                output = output[0]  # 배치 차원 제거
            
            # 신뢰도 임계값 필터링
            confidences = output[:, 4]
            valid_indices = confidences > conf_threshold
            
            if np.any(valid_indices):
                valid_outputs = output[valid_indices]
                
                for det in valid_outputs:
                    x_center, y_center, width, height, conf, class_id = det[:6]
                    
                    detections.append({
                        'bbox': [x_center, y_center, width, height],
                        'confidence': float(conf),
                        'class_id': int(class_id)
                    })
        
        return detections
    
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Tuple[List[Dict], float]:
        """추론 실행 및 성능 측정"""
        # 이미지 전처리
        img_batch, original_shape = self.preprocess_image(image_path)
        
        # CPU 사용률 측정 시작
        cpu_percent_start = psutil.cpu_percent(interval=0.1)
        
        # 추론 시간 측정
        start_time = time.time()
        
        # ONNX Runtime 추론
        outputs = self.session.run(self.output_names, {self.input_name: img_batch})
        
        # 추론 완료
        inference_time = time.time() - start_time
        
        # CPU 사용률 측정
        cpu_percent_end = psutil.cpu_percent(interval=0.1)
        cpu_usage = max(cpu_percent_start, cpu_percent_end)
        
        # 메모리 사용량
        memory_info = psutil.virtual_memory()
        memory_used_mb = memory_info.used / (1024 ** 2)
        
        # 후처리
        detections = self.postprocess_output(outputs, conf_threshold)
        
        return detections, inference_time, cpu_usage, memory_used_mb
    
    def test_inference(self, image_paths: List[str], num_runs: int = 5):
        """추론 성능 테스트"""
        print("\n" + "="*60)
        print("🧪 라즈베리파이 시뮬레이션 테스트 시작")
        print("="*60)
        
        all_times = []
        all_cpu_usage = []
        all_memory_usage = []
        
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_path}")
                continue
            
            print(f"\n📸 테스트 이미지: {img_path.name}")
            
            times = []
            cpu_usages = []
            memory_usages = []
            
            # 워밍업 (첫 실행은 느릴 수 있음)
            print("   워밍업 중...")
            self.predict(str(img_path))
            
            # 실제 측정
            print(f"   {num_runs}회 추론 실행 중...")
            for i in range(num_runs):
                detections, inference_time, cpu_usage, memory_used = self.predict(str(img_path))
                times.append(inference_time)
                cpu_usages.append(cpu_usage)
                memory_usages.append(memory_used)
                
                if i == 0:
                    print(f"   첫 번째 추론: {inference_time*1000:.1f}ms, 감지된 객체: {len(detections)}개")
            
            avg_time = np.mean(times[1:])  # 첫 번째 제외한 평균
            min_time = np.min(times[1:])
            max_time = np.max(times[1:])
            avg_cpu = np.mean(cpu_usages[1:])
            avg_memory = np.mean(memory_usages[1:])
            
            print(f"   평균 추론 시간: {avg_time*1000:.1f}ms (최소: {min_time*1000:.1f}ms, 최대: {max_time*1000:.1f}ms)")
            print(f"   평균 CPU 사용률: {avg_cpu:.1f}%")
            print(f"   평균 메모리 사용량: {avg_memory:.1f}MB")
            print(f"   예상 FPS: {1/avg_time:.1f}")
            
            all_times.extend(times[1:])
            all_cpu_usage.extend(cpu_usages[1:])
            all_memory_usage.extend(memory_usages[1:])
        
        # 전체 통계
        print("\n" + "="*60)
        print("📊 전체 성능 통계")
        print("="*60)
        print(f"전체 평균 추론 시간: {np.mean(all_times)*1000:.1f}ms")
        print(f"전체 평균 CPU 사용률: {np.mean(all_cpu_usage):.1f}%")
        print(f"전체 평균 메모리 사용량: {np.mean(all_memory_usage):.1f}MB")
        print(f"전체 평균 FPS: {1/np.mean(all_times):.1f}")
        
        # 라즈베리파이 성능 평가
        print("\n" + "="*60)
        print("🎯 라즈베리파이 호환성 평가")
        print("="*60)
        
        avg_time_ms = np.mean(all_times) * 1000
        avg_fps = 1 / np.mean(all_times)
        
        if avg_time_ms < 100:
            print("✅ 매우 빠름: 실시간 처리 가능 (목표: 15 FPS 이상)")
        elif avg_time_ms < 200:
            print("✅ 빠름: 실시간 처리 가능 (목표: 5-15 FPS)")
        elif avg_time_ms < 500:
            print("⚠️ 보통: 실시간 처리는 어려울 수 있음 (목표: 2-5 FPS)")
        else:
            print("❌ 느림: 실시간 처리 어려움 (목표: 2 FPS 미만)")
        
        print(f"\n권장 사항:")
        if avg_time_ms > 200:
            print("   - 이미지 크기를 줄이는 것을 고려하세요 (640 -> 416)")
            print("   - 입력 전처리를 최적화하세요")
            print("   - ONNX 모델을 더 단순화하세요")
        else:
            print("   - 현재 설정으로 라즈베리파이에서 사용 가능합니다")
        
        return {
            'avg_inference_time_ms': avg_time_ms,
            'avg_fps': avg_fps,
            'avg_cpu_usage': np.mean(all_cpu_usage),
            'avg_memory_mb': np.mean(all_memory_usage)
        }


def export_to_onnx(model_path: str, output_path: str = None):
    """모델을 ONNX로 변환"""
    from ultralytics import YOLO
    
    print(f"📤 ONNX 변환 시작: {model_path}")
    
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # ONNX 변환 (라즈베리파이 최적화)
    exported_path = model.export(
        format='onnx',
        imgsz=640,
        dynamic=False,  # 정적 크기 (더 빠름)
        simplify=True,  # 모델 단순화
        opset=12  # ONNX opset 버전 (호환성)
    )
    
    print(f"✅ ONNX 변환 완료: {exported_path}")
    return exported_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="라즈베리파이 배포 전 테스트")
    parser.add_argument("--model", default="results/training_20251031_201041/best_model.pt",
                       help="학습된 모델 경로 (.pt 또는 .onnx)")
    parser.add_argument("--images", nargs="+", 
                       default=["../real_image0.jpeg", "../real_image1.jpeg"],
                       help="테스트 이미지 경로")
    parser.add_argument("--export", action="store_true",
                       help="먼저 ONNX로 변환")
    parser.add_argument("--runs", type=int, default=5,
                       help="각 이미지당 추론 실행 횟수")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    # ONNX 변환
    if args.export or not model_path.suffix == '.onnx':
        if model_path.suffix == '.pt':
            print("🔄 .pt 파일을 ONNX로 변환합니다...")
            onnx_path = export_to_onnx(str(model_path))
        else:
            onnx_path = str(model_path)
    else:
        onnx_path = str(model_path)
    
    # 라즈베리파이 시뮬레이션 테스트
    simulator = RaspberryPiSimulator(onnx_path)
    results = simulator.test_inference(args.images, num_runs=args.runs)
    
    print("\n✅ 테스트 완료!")


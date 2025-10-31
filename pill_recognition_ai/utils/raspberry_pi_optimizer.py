"""
라즈베리파이 4 Model B용 모델 최적화 및 추론 모듈
RTX 3060Ti에서 학습한 모델을 라즈베리파이에서 실행 가능하도록 최적화합니다.
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import psutil
import subprocess
import platform


class RaspberryPiOptimizer:
    """라즈베리파이 4 Model B 최적화 클래스"""
    
    def __init__(self, config_path: str = "configs/raspberry_pi_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.pi_config = self.config['raspberry_pi']
        self.inference_config = self.config['inference']
        
        # 라즈베리파이 환경 감지
        self.is_raspberry_pi = self._detect_raspberry_pi()
        
        if self.is_raspberry_pi:
            print("🍓 라즈베리파이 환경 감지됨")
            self._setup_raspberry_pi_environment()
        else:
            print("💻 일반 PC 환경 (라즈베리파이 최적화 모드)")
    
    def _detect_raspberry_pi(self) -> bool:
        """라즈베리파이 환경 감지"""
        try:
            # CPU 정보 확인
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo:
                    return True
            
            # 시스템 정보 확인
            if 'arm' in platform.machine().lower():
                return True
                
        except:
            pass
        
        return False
    
    def _setup_raspberry_pi_environment(self):
        """라즈베리파이 환경 설정"""
        if not self.is_raspberry_pi:
            return
        
        print("⚙️ 라즈베리파이 환경 설정 중...")
        
        # CPU 주파수 설정
        try:
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'ondemand'], 
                         check=True, capture_output=True)
            print("✅ CPU 주파수 조절기 설정 완료")
        except:
            print("⚠️ CPU 주파수 설정 실패 (권한 필요)")
        
        # 메모리 설정 확인
        self._check_memory_settings()
        
        # 온도 모니터링 시작
        self._start_temperature_monitoring()
    
    def _check_memory_settings(self):
        """메모리 설정 확인"""
        try:
            # 스왑 설정 확인
            with open('/proc/swaps', 'r') as f:
                swaps = f.read()
                if 'swap' not in swaps.lower():
                    print("⚠️ 스왑이 활성화되지 않았습니다. 메모리 부족 시 성능 저하 가능")
        except:
            pass
    
    def _start_temperature_monitoring(self):
        """온도 모니터링 시작"""
        if not self.is_raspberry_pi:
            return
        
        try:
            # 온도 파일 경로 확인
            temp_files = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/hwmon/hwmon0/temp1_input'
            ]
            
            self.temp_file = None
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    self.temp_file = temp_file
                    break
            
            if self.temp_file:
                print(f"🌡️ 온도 모니터링 활성화: {self.temp_file}")
            else:
                print("⚠️ 온도 센서를 찾을 수 없습니다")
                
        except Exception as e:
            print(f"⚠️ 온도 모니터링 설정 실패: {e}")
    
    def get_system_info(self) -> Dict:
        """시스템 정보 반환"""
        info = {
            'platform': platform.platform(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'memory_available': psutil.virtual_memory().available / 1024**3,  # GB
            'is_raspberry_pi': self.is_raspberry_pi
        }
        
        if self.is_raspberry_pi and hasattr(self, 'temp_file') and self.temp_file:
            try:
                with open(self.temp_file, 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    info['temperature'] = temp_millicelsius / 1000  # 섭씨
            except:
                info['temperature'] = None
        
        return info
    
    def optimize_model_for_raspberry_pi(self, model_path: str) -> Dict:
        """라즈베리파이용 모델 최적화"""
        print("🍓 라즈베리파이용 모델 최적화 시작...")
        
        try:
            from ultralytics import YOLO
            
            # 모델 로드
            model = YOLO(model_path)
            
            # 최적화 설정
            optimization_config = self.pi_config['model_optimization']
            
            exported_models = {}
            
            # ONNX 변환 (라즈베리파이에서 가장 호환성 좋음)
            if optimization_config.get('quantization', False):
                print("🔄 INT8 양자화 ONNX 변환 중...")
                try:
                    onnx_path = model.export(
                        format='onnx',
                        imgsz=self.config['data']['inference_image_size'],
                        int8=True,  # INT8 양자화
                        dynamic=False,  # 정적 배치 크기
                        simplify=True,  # 모델 단순화
                        opset=11  # ONNX opset 버전
                    )
                    exported_models['onnx_int8'] = onnx_path
                    print(f"✅ INT8 양자화 ONNX 저장: {onnx_path}")
                except Exception as e:
                    print(f"⚠️ INT8 양자화 실패: {e}")
            
            # 일반 ONNX 변환
            print("🔄 일반 ONNX 변환 중...")
            onnx_path = model.export(
                format='onnx',
                imgsz=self.config['data']['inference_image_size'],
                dynamic=False,
                simplify=True,
                opset=11
            )
            exported_models['onnx'] = onnx_path
            print(f"✅ ONNX 저장: {onnx_path}")
            
            # TensorFlow Lite 변환 (모바일 최적화)
            print("🔄 TensorFlow Lite 변환 중...")
            try:
                tflite_path = model.export(
                    format='tflite',
                    imgsz=self.config['data']['inference_image_size'],
                    int8=True,  # INT8 양자화
                    dynamic=False
                )
                exported_models['tflite'] = tflite_path
                print(f"✅ TensorFlow Lite 저장: {tflite_path}")
            except Exception as e:
                print(f"⚠️ TensorFlow Lite 변환 실패: {e}")
            
            # 모델 크기 정보
            for format_name, model_path in exported_models.items():
                size_mb = os.path.getsize(model_path) / 1024**2
                print(f"📊 {format_name} 모델 크기: {size_mb:.1f}MB")
            
            return exported_models
            
        except Exception as e:
            print(f"❌ 모델 최적화 실패: {e}")
            return {}
    
    def benchmark_model_performance(self, model_path: str, test_images: List[str]) -> Dict:
        """모델 성능 벤치마크"""
        print("📊 라즈베리파이 모델 성능 벤치마크 시작...")
        
        try:
            import onnxruntime as ort
            
            # ONNX Runtime 세션 생성
            providers = ['CPUExecutionProvider']  # 라즈베리파이에서는 CPU만 사용
            
            session = ort.InferenceSession(model_path, providers=providers)
            
            # 입력 크기 확인
            input_shape = session.get_inputs()[0].shape
            input_height, input_width = input_shape[2], input_shape[3]
            
            print(f"📐 모델 입력 크기: {input_width}x{input_height}")
            
            # 성능 측정
            inference_times = []
            memory_usage = []
            
            for i, img_path in enumerate(test_images[:10]):  # 최대 10개 이미지로 테스트
                # 이미지 전처리
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (input_width, input_height))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)
                
                # 메모리 사용량 측정 시작
                memory_before = psutil.virtual_memory().used / 1024**2  # MB
                
                # 추론 실행
                start_time = time.time()
                outputs = session.run(None, {session.get_inputs()[0].name: img_batch})
                inference_time = time.time() - start_time
                
                # 메모리 사용량 측정 종료
                memory_after = psutil.virtual_memory().used / 1024**2  # MB
                
                inference_times.append(inference_time)
                memory_usage.append(memory_after - memory_before)
                
                print(f"   이미지 {i+1}: {inference_time*1000:.1f}ms, 메모리: {memory_usage[-1]:.1f}MB")
            
            # 통계 계산
            avg_inference_time = np.mean(inference_times)
            avg_fps = 1.0 / avg_inference_time
            avg_memory = np.mean(memory_usage)
            
            benchmark_results = {
                'avg_inference_time_ms': avg_inference_time * 1000,
                'avg_fps': avg_fps,
                'avg_memory_usage_mb': avg_memory,
                'max_memory_usage_mb': np.max(memory_usage),
                'model_size_mb': os.path.getsize(model_path) / 1024**2,
                'input_size': f"{input_width}x{input_height}",
                'test_images': len(test_images)
            }
            
            print(f"\n📈 벤치마크 결과:")
            print(f"   평균 추론 시간: {benchmark_results['avg_inference_time_ms']:.1f}ms")
            print(f"   평균 FPS: {benchmark_results['avg_fps']:.1f}")
            print(f"   평균 메모리 사용량: {benchmark_results['avg_memory_usage_mb']:.1f}MB")
            print(f"   모델 크기: {benchmark_results['model_size_mb']:.1f}MB")
            
            # 목표 성능과 비교
            targets = self.pi_config['performance_targets']
            print(f"\n🎯 목표 성능 비교:")
            print(f"   FPS: {benchmark_results['avg_fps']:.1f} / {targets['fps']} (목표)")
            print(f"   지연시간: {benchmark_results['avg_inference_time_ms']:.1f}ms / {targets['latency_ms']}ms (목표)")
            print(f"   메모리: {benchmark_results['avg_memory_usage_mb']:.1f}MB / {targets['memory_usage_mb']}MB (목표)")
            
            return benchmark_results
            
        except Exception as e:
            print(f"❌ 벤치마크 실패: {e}")
            return {}
    
    def create_raspberry_pi_deployment_package(self, optimized_models: Dict, 
                                             config_path: str = "raspberry_pi_deployment") -> str:
        """라즈베리파이 배포 패키지 생성"""
        print("📦 라즈베리파이 배포 패키지 생성 중...")
        
        deployment_dir = Path(config_path)
        deployment_dir.mkdir(exist_ok=True)
        
        # 추론 스크립트 생성
        inference_script = self._create_inference_script()
        script_path = deployment_dir / "pill_inference.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(inference_script)
        
        # 설정 파일 복사
        config_file = deployment_dir / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # 최적화된 모델들 복사
        models_dir = deployment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for format_name, model_path in optimized_models.items():
            dest_path = models_dir / f"pill_model.{format_name.split('_')[0]}"
            import shutil
            shutil.copy2(model_path, dest_path)
            print(f"✅ 모델 복사: {dest_path}")
        
        # requirements.txt 생성
        requirements = [
            "onnxruntime>=1.15.0",
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "pyyaml>=6.0",
            "psutil>=5.9.0"
        ]
        
        req_path = deployment_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        # README 생성
        readme_content = self._create_deployment_readme()
        readme_path = deployment_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ 배포 패키지 생성 완료: {deployment_dir}")
        return str(deployment_dir)
    
    def _create_inference_script(self) -> str:
        """라즈베리파이용 추론 스크립트 생성"""
        return '''"""
라즈베리파이 4 Model B용 알약 인식 추론 스크립트
"""

import cv2
import numpy as np
import yaml
import time
import psutil
from pathlib import Path
import onnxruntime as ort


class PillInference:
    """라즈베리파이용 알약 인식 추론 클래스"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['data']['classes']
        self.confidence_threshold = self.config['inference']['postprocessing']['confidence_threshold']
        self.nms_threshold = self.config['inference']['postprocessing']['nms_threshold']
        
        # 모델 로드
        self.model_path = "models/pill_model.onnx"
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        
        # 입력 크기 확인
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        
        print(f"🍓 라즈베리파이 알약 인식 시스템 초기화 완료")
        print(f"📐 입력 크기: {self.input_width}x{self.input_height}")
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 크기 조정
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # 색상 변환 및 정규화
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # 배치 차원 추가 및 채널 순서 변경
        img_batch = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)
        
        return img_batch
    
    def postprocess_outputs(self, outputs, original_shape):
        """모델 출력 후처리"""
        # YOLO 출력 파싱 (간단한 버전)
        predictions = outputs[0][0]  # 첫 번째 배치
        
        # 신뢰도 임계값 필터링
        confidences = predictions[:, 4]
        mask = confidences > self.confidence_threshold
        filtered_predictions = predictions[mask]
        
        if len(filtered_predictions) == 0:
            return []
        
        # 바운딩 박스 좌표 변환
        boxes = []
        scores = []
        class_ids = []
        
        for pred in filtered_predictions:
            x_center, y_center, width, height = pred[:4]
            confidence = pred[4]
            class_scores = pred[5:]
            
            # 가장 높은 점수의 클래스 선택
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # 최종 신뢰도 계산
            final_confidence = confidence * class_score
            
            if final_confidence > self.confidence_threshold:
                # 절대 좌표로 변환
                orig_h, orig_w = original_shape[:2]
                x1 = int((x_center - width/2) * orig_w)
                y1 = int((y_center - height/2) * orig_h)
                x2 = int((x_center + width/2) * orig_w)
                y2 = int((y_center + height/2) * orig_h)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(final_confidence)
                class_ids.append(class_id)
        
        # NMS 적용
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)
            
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                      self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [(boxes[i], scores[i], class_ids[i]) for i in indices]
        
        return []
    
    def predict(self, image):
        """이미지에서 알약 예측"""
        start_time = time.time()
        
        # 전처리
        input_tensor = self.preprocess_image(image)
        
        # 추론
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        
        # 후처리
        detections = self.postprocess_outputs(outputs, image.shape)
        
        inference_time = time.time() - start_time
        
        # 결과 정리
        results = []
        for box, score, class_id in detections:
            results.append({
                'bbox': box,
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': self.classes[class_id]
            })
        
        return results, inference_time
    
    def draw_results(self, image, results):
        """결과 시각화"""
        img_with_boxes = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            confidence = result['confidence']
            class_name = result['class_name']
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 그리기
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img_with_boxes


def main():
    """메인 함수"""
    # 추론 시스템 초기화
    inference_system = PillInference()
    
    # 카메라 또는 이미지 파일로 테스트
    cap = cv2.VideoCapture(0)  # 웹캠 사용
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 이미지 파일 모드로 전환합니다.")
        # 이미지 파일 테스트
        test_image_path = "test_image.jpg"
        if Path(test_image_path).exists():
            image = cv2.imread(test_image_path)
            results, inference_time = inference_system.predict(image)
            
            print(f"🔍 감지된 알약: {len(results)}개")
            print(f"⏱️ 추론 시간: {inference_time*1000:.1f}ms")
            
            for i, result in enumerate(results):
                print(f"   {i+1}. {result['class_name']} (신뢰도: {result['confidence']:.3f})")
            
            # 결과 시각화
            img_with_results = inference_system.draw_results(image, results)
            cv2.imwrite("result.jpg", img_with_results)
            print("✅ 결과 이미지 저장: result.jpg")
        else:
            print("❌ 테스트 이미지 파일을 찾을 수 없습니다.")
        return
    
    print("📹 실시간 추론 시작 (ESC 키로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 추론 실행
        results, inference_time = inference_system.predict(frame)
        
        # 결과 시각화
        frame_with_results = inference_system.draw_results(frame, results)
        
        # 정보 표시
        info_text = f"FPS: {1/inference_time:.1f} | Pills: {len(results)}"
        cv2.putText(frame_with_results, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 화면 출력
        cv2.imshow('Pill Recognition', frame_with_results)
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
    
    def _create_deployment_readme(self) -> str:
        """배포용 README 생성"""
        return '''# 라즈베리파이 4 Model B 알약 인식 시스템

## 설치 방법

### 1. 시스템 업데이트
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Python 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
python pill_inference.py
```

## 성능 목표
- FPS: 5 FPS 이상
- 지연시간: 200ms 이하
- 메모리 사용량: 512MB 이하

## 문제 해결
- 메모리 부족 시: 스왑 활성화
- 온도 과열 시: 쿨링 팬 설치
- 성능 저하 시: CPU 주파수 확인

## 하드웨어 요구사항
- 라즈베리파이 4 Model B (4GB RAM 권장)
- MicroSD 카드 (32GB 이상)
- 웹캠 또는 카메라 모듈
'''


if __name__ == "__main__":
    # 라즈베리파이 최적화 테스트
    optimizer = RaspberryPiOptimizer()
    
    # 시스템 정보 출력
    system_info = optimizer.get_system_info()
    print("🍓 시스템 정보:")
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # 모델 최적화 (예시)
    # model_path = "results/training_20241027_085100/best_model.pt"
    # optimized_models = optimizer.optimize_model_for_raspberry_pi(model_path)
    
    # 배포 패키지 생성
    # deployment_dir = optimizer.create_raspberry_pi_deployment_package(optimized_models)

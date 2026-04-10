"""
RTX 3060Ti 8GB VRAM 최적화 모듈
GPU 메모리 관리 및 성능 최적화를 위한 유틸리티
"""

import os
import torch
import gc
import psutil
import GPUtil
from typing import Dict, List, Optional
import warnings


class GPUOptimizer:
    """RTX 3060Ti 8GB VRAM 최적화 클래스"""
    
    def __init__(self):
        """초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_memory_total = 0
        self.gpu_memory_used = 0
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"🎮 GPU 감지: {torch.cuda.get_device_name(0)}")
            print(f"💾 총 VRAM: {self.gpu_memory_total:.1f}GB")
        
        # 메모리 최적화 설정
        self._setup_memory_optimization()
    
    def _setup_memory_optimization(self):
        """메모리 최적화 설정"""
        if torch.cuda.is_available():
            # CUDA 메모리 할당 전략 설정
            torch.cuda.empty_cache()
            
            # 메모리 증가 허용 (단계적 할당)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # 메모리 풀링 활성화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            print("✅ GPU 메모리 최적화 설정 완료")
    
    def get_memory_info(self) -> Dict:
        """현재 메모리 사용량 정보 반환"""
        memory_info = {
            'cpu_memory': {
                'total': psutil.virtual_memory().total / 1024**3,
                'used': psutil.virtual_memory().used / 1024**3,
                'available': psutil.virtual_memory().available / 1024**3,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_memory'] = {
                'total': self.gpu_memory_total,
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'cached': torch.cuda.memory_reserved() / 1024**3,
                'free': (self.gpu_memory_total * 1024**3 - torch.cuda.memory_reserved()) / 1024**3
            }
            
            # GPU 온도 및 사용률 정보
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_info['gpu_memory']['temperature'] = gpu.temperature
                    memory_info['gpu_memory']['load'] = gpu.load * 100
            except:
                pass
        
        return memory_info
    
    def print_memory_status(self):
        """메모리 상태 출력"""
        memory_info = self.get_memory_info()
        
        print("\n📊 메모리 사용량:")
        print(f"   CPU: {memory_info['cpu_memory']['used']:.1f}GB / {memory_info['cpu_memory']['total']:.1f}GB ({memory_info['cpu_memory']['percent']:.1f}%)")
        
        if 'gpu_memory' in memory_info:
            gpu_mem = memory_info['gpu_memory']
            print(f"   GPU: {gpu_mem['allocated']:.1f}GB / {gpu_mem['total']:.1f}GB (사용률: {gpu_mem['allocated']/gpu_mem['total']*100:.1f}%)")
            print(f"   GPU 캐시: {gpu_mem['cached']:.1f}GB")
            print(f"   GPU 여유: {gpu_mem['free']:.1f}GB")
            
            if 'temperature' in gpu_mem:
                print(f"   GPU 온도: {gpu_mem['temperature']}°C")
            if 'load' in gpu_mem:
                print(f"   GPU 사용률: {gpu_mem['load']:.1f}%")
    
    def clear_memory(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        print("🧹 메모리 정리 완료")
    
    def get_optimal_batch_size(self, model, input_size: tuple = (3, 640, 640), 
                              max_batch_size: int = 16, safety_margin: float = 0.8) -> int:
        """최적 배치 크기 자동 탐지"""
        if not torch.cuda.is_available():
            return 1
        
        print("🔍 최적 배치 크기 탐지 중...")
        
        # 메모리 정리
        self.clear_memory()
        
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                # 테스트 입력 생성
                test_input = torch.randn(batch_size, *input_size).to(self.device)
                
                # 모델을 GPU로 이동
                model = model.to(self.device)
                
                # 순전파 테스트
                with torch.no_grad():
                    _ = model(test_input)
                
                # 메모리 사용량 확인
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_ratio = memory_used / self.gpu_memory_total
                
                if memory_ratio < safety_margin:
                    optimal_batch_size = batch_size
                    print(f"   배치 크기 {batch_size}: 메모리 사용률 {memory_ratio*100:.1f}% ✅")
                else:
                    print(f"   배치 크기 {batch_size}: 메모리 사용률 {memory_ratio*100:.1f}% ❌ (초과)")
                    break
                
                # 메모리 정리
                del test_input
                self.clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   배치 크기 {batch_size}: OOM 에러 ❌")
                    break
                else:
                    raise e
        
        print(f"🎯 최적 배치 크기: {optimal_batch_size}")
        return optimal_batch_size
    
    def optimize_model_for_inference(self, model):
        """추론용 모델 최적화"""
        if torch.cuda.is_available():
            # 모델을 GPU로 이동
            model = model.to(self.device)
            
            # 평가 모드로 설정
            model.eval()
            
            # 추론 최적화
            with torch.no_grad():
                # JIT 컴파일 (PyTorch 1.8+)
                try:
                    model = torch.jit.optimize_for_inference(model)
                    print("✅ JIT 추론 최적화 완료")
                except:
                    print("⚠️ JIT 최적화 실패 (PyTorch 버전 확인 필요)")
            
            # 메모리 정리
            self.clear_memory()
        
        return model
    
    def monitor_training_memory(self, model, dataloader, max_iterations: int = 10):
        """학습 중 메모리 모니터링"""
        print("📈 학습 중 메모리 모니터링 시작...")
        
        model = model.to(self.device)
        model.train()
        
        memory_usage = []
        
        for i, batch in enumerate(dataloader):
            if i >= max_iterations:
                break
            
            try:
                # 배치를 GPU로 이동
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) if torch.is_tensor(item) else item for item in batch]
                else:
                    batch = batch.to(self.device)
                
                # 순전파
                outputs = model(batch)
                
                # 메모리 사용량 기록
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_usage.append(memory_used)
                
                print(f"   반복 {i+1}: {memory_used:.2f}GB 사용")
                
                # 메모리 정리
                del outputs
                if i % 5 == 0:  # 5번마다 메모리 정리
                    self.clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ 반복 {i+1}에서 OOM 에러 발생")
                    break
                else:
                    raise e
        
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
            print(f"\n📊 메모리 사용량 통계:")
            print(f"   평균: {avg_memory:.2f}GB")
            print(f"   최대: {max_memory:.2f}GB")
            print(f"   GPU 사용률: {max_memory/self.gpu_memory_total*100:.1f}%")
        
        return memory_usage
    
    def get_recommended_settings(self) -> Dict:
        """RTX 3060Ti 권장 설정 반환"""
        settings = {
            'model_size': 'yolov8n',  # 가장 가벼운 모델 권장
            'batch_size': 8,
            'image_size': 640,
            'workers': 4,
            'mixed_precision': True,  # AMP 사용 권장
            'gradient_accumulation': 2,  # 배치 크기 효과적 증가
            'memory_optimization': {
                'empty_cache_frequency': 10,  # 10 에포크마다 캐시 정리
                'gradient_checkpointing': True,
                'pin_memory': True
            }
        }
        
        print("💡 RTX 3060Ti 권장 설정:")
        for key, value in settings.items():
            print(f"   {key}: {value}")
        
        return settings


if __name__ == "__main__":
    # GPU 최적화 테스트
    optimizer = GPUOptimizer()
    optimizer.print_memory_status()
    
    # 권장 설정 출력
    settings = optimizer.get_recommended_settings()
    
    # 메모리 정리
    optimizer.clear_memory()

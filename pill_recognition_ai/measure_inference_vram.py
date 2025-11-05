"""
추론 시 VRAM 사용량 측정 스크립트
"""
import torch
from ultralytics import YOLO
from pathlib import Path
from utils.gpu_optimizer import GPUOptimizer
import gc

def measure_inference_vram(model_path: str, image_paths: list):
    """추론 시 VRAM 사용량 측정"""
    optimizer = GPUOptimizer()
    
    print("\n" + "="*60)
    print("📊 추론 전 메모리 상태")
    print("="*60)
    optimizer.print_memory_status()
    
    # 메모리 정리
    optimizer.clear_memory()
    
    print("\n" + "="*60)
    print("📦 모델 로드 중...")
    print("="*60)
    
    # 모델 로드 전 메모리
    before_load = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 모델 로드 후 메모리
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    after_load = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    model_load_vram = after_load - before_load
    
    print(f"✅ 모델 로드 완료")
    print(f"   모델 로드로 인한 VRAM 증가: {model_load_vram:.2f}GB")
    
    print("\n" + "="*60)
    print("🔍 추론 중 메모리 모니터링")
    print("="*60)
    
    max_vram_used = 0
    inference_vram_usage = []
    
    for idx, img_path in enumerate(image_paths, 1):
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_path}")
            continue
        
        # 추론 전 메모리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before_inference = torch.cuda.memory_allocated() / 1024**3
        else:
            before_inference = 0
        
        print(f"\n📸 이미지 {idx}: {img_path.name}")
        print(f"   추론 전 VRAM: {before_inference:.2f}GB")
        
        # 추론 실행
        results = model.predict(str(img_path), imgsz=640, conf=0.25, verbose=False)
        
        # 추론 후 메모리
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            after_inference = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            after_inference = 0
            peak_memory = 0
        
        inference_vram = after_inference - before_inference
        inference_vram_usage.append(inference_vram)
        max_vram_used = max(max_vram_used, peak_memory)
        
        print(f"   추론 후 VRAM: {after_inference:.2f}GB")
        print(f"   추론으로 인한 VRAM 증가: {inference_vram:.2f}GB")
        print(f"   피크 VRAM: {peak_memory:.2f}GB")
        
        # 감지된 객체 수
        if results and len(results) > 0:
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"   감지된 알약 수: {detections}개")
        
        # 메모리 정리 (다음 이미지를 위해)
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("📊 최종 메모리 사용량 요약")
    print("="*60)
    
    if torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated() / 1024**3
        final_cached = torch.cuda.memory_reserved() / 1024**3
        
        print(f"모델 로드 VRAM: {model_load_vram:.2f}GB")
        if inference_vram_usage:
            avg_inference_vram = sum(inference_vram_usage) / len(inference_vram_usage)
            print(f"평균 추론 VRAM 증가: {avg_inference_vram:.2f}GB")
            print(f"최대 추론 VRAM (피크): {max_vram_used:.2f}GB")
        print(f"현재 할당된 VRAM: {final_vram:.2f}GB")
        print(f"현재 캐시된 VRAM: {final_cached:.2f}GB")
        print(f"총 VRAM 사용률: {(final_cached / optimizer.gpu_memory_total * 100):.1f}%")
        
        # 권장 사항
        print("\n💡 권장 사항:")
        if final_cached > optimizer.gpu_memory_total * 0.8:
            print("   ⚠️ VRAM 사용률이 80% 이상입니다. 배치 크기를 줄이거나 이미지 크기를 줄이는 것을 고려하세요.")
        elif final_cached > optimizer.gpu_memory_total * 0.6:
            print("   ✅ VRAM 사용률이 적절합니다. 현재 설정으로 사용 가능합니다.")
        else:
            print("   ✅ VRAM 사용률이 낮습니다. 배치 크기를 늘릴 수 있습니다.")
    
    # 메모리 정리
    optimizer.clear_memory()
    
    return {
        'model_load_vram': model_load_vram,
        'inference_vram_usage': inference_vram_usage,
        'max_inference_vram': max_vram_used,
        'final_vram': final_vram if torch.cuda.is_available() else 0,
        'final_cached': final_cached if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    # 측정할 모델과 이미지 경로
    model_path = "results/training_20251031_201041/best_model.pt"
    image_paths = [
        "../real_image0.jpeg",
        "../real_image1.jpeg"
    ]
    
    # VRAM 사용량 측정
    results = measure_inference_vram(model_path, image_paths)
    
    print("\n✅ VRAM 측정 완료!")


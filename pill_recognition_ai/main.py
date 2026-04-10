"""
알약 인식 AI 모델 개발 메인 실행 스크립트

"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_preprocessing import PillDataProcessor
from utils.data_augmentation import PillDataAugmenter
from utils.gpu_optimizer import GPUOptimizer
from utils.visualizer import ResultVisualizer
from models.yolo_trainer import PillModelTrainer
from models.model_evaluator import ModelEvaluator


class PillRecognitionPipeline:
    """알약 인식 AI 모델 개발 전체 파이프라인"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """초기화"""
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # GPU 최적화 설정
        self.gpu_optimizer = GPUOptimizer()
        self.gpu_optimizer.print_memory_status()
        
        print("🚀 알약 인식 AI 모델 개발 파이프라인 초기화 완료")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"💾 VRAM: {self.gpu_optimizer.gpu_memory_total:.1f}GB")
    
    def step1_data_preprocessing(self):
        """1단계: 데이터 전처리"""
        print("\n" + "="*60)
        print("📊 1단계: 데이터 전처리 시작")
        print("="*60)
        
        processor = PillDataProcessor(self.config_path)
        yolo_config_path = processor.process_all_data()
        
        print("✅ 데이터 전처리 완료!")
        return yolo_config_path
    
    def step2_data_augmentation(self, dataset_path: str = "dataset"):
        """2단계: 데이터 증강"""
        print("\n" + "="*60)
        print("🔄 2단계: 데이터 증강 시작")
        print("="*60)
        
        augmenter = PillDataAugmenter(self.config_path)
        augmenter.augment_dataset(dataset_path, "dataset_augmented", augmentation_factor=2)
        
        print("✅ 데이터 증강 완료!")
        return "dataset_augmented"
    
    def step3_model_training(self, dataset_yaml_path: str):
        """3단계: 모델 학습"""
        print("\n" + "="*60)
        print("🎯 3단계: 모델 학습 시작")
        print("="*60)
        
        # GPU 메모리 정리
        self.gpu_optimizer.clear_memory()
        
        trainer = PillModelTrainer(self.config_path)
        results = trainer.train_model(dataset_yaml_path)
        
        print("✅ 모델 학습 완료!")
        return results
    
    def step4_model_evaluation(self, model_path: str, dataset_yaml_path: str):
        """4단계: 모델 평가"""
        print("\n" + "="*60)
        print("🔍 4단계: 모델 평가 시작")
        print("="*60)
        
        evaluator = ModelEvaluator(self.config_path)
        results = evaluator.evaluate_model(model_path, dataset_yaml_path)
        
        print("✅ 모델 평가 완료!")
        return results
    
    def step5_model_optimization(self, model_path: str):
        """5단계: 모델 최적화 (ONNX 변환)"""
        print("\n" + "="*60)
        print("⚡ 5단계: 모델 최적화 시작")
        print("="*60)
        
        trainer = PillModelTrainer(self.config_path)
        exported_models = trainer.export_model(model_path, ['onnx'])
        
        print("✅ 모델 최적화 완료!")
        return exported_models
    
    def run_full_pipeline(self, skip_steps: list = None):
        """전체 파이프라인 실행"""
        if skip_steps is None:
            skip_steps = []
        
        print("🚀 알약 인식 AI 모델 개발 전체 파이프라인 시작!")
        print(f"⏭️ 건너뛸 단계: {skip_steps}")
        
        try:
            # 1단계: 데이터 전처리
            if 1 not in skip_steps:
                yolo_config_path = self.step1_data_preprocessing()
            else:
                yolo_config_path = "dataset/pill_dataset.yaml"
                print("⏭️ 1단계 건너뛰기: 데이터 전처리")
            
            # 2단계: 데이터 증강
            if 2 not in skip_steps:
                augmented_dataset = self.step2_data_augmentation()
                yolo_config_path = f"{augmented_dataset}/pill_dataset.yaml"
            else:
                print("⏭️ 2단계 건너뛰기: 데이터 증강")
            
            # 3단계: 모델 학습
            if 3 not in skip_steps:
                training_results = self.step3_model_training(yolo_config_path)
                model_path = training_results.save_dir / "weights" / "best.pt"
            else:
                # 기존 모델 찾기
                results_dir = Path("results")
                model_paths = list(results_dir.glob("*/best_model.pt"))
                if model_paths:
                    model_path = model_paths[-1]  # 가장 최근 모델
                    print(f"⏭️ 3단계 건너뛰기: 기존 모델 사용 - {model_path}")
                else:
                    print("❌ 학습된 모델을 찾을 수 없습니다!")
                    return
            
            # 4단계: 모델 평가
            if 4 not in skip_steps:
                evaluation_results = self.step4_model_evaluation(str(model_path), yolo_config_path)
            else:
                print("⏭️ 4단계 건너뛰기: 모델 평가")
            
            # 5단계: 모델 최적화
            if 5 not in skip_steps:
                exported_models = self.step5_model_optimization(str(model_path))
            else:
                print("⏭️ 5단계 건너뛰기: 모델 최적화")
            
            print("\n" + "="*60)
            print("🎉 전체 파이프라인 완료!")
            print("="*60)
            
            # 최종 메모리 상태 출력
            self.gpu_optimizer.print_memory_status()
            
        except Exception as e:
            print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def run_quick_test(self):
        """빠른 테스트 실행 (데이터 전처리만)"""
        print("🧪 빠른 테스트 모드 실행...")
        
        try:
            # 데이터 전처리만 실행
            yolo_config_path = self.step1_data_preprocessing()
            
            print("✅ 빠른 테스트 완료!")
            print(f"📁 생성된 데이터셋: {yolo_config_path}")
            
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="알약 인식 AI 모델 개발 파이프라인")
    parser.add_argument("--mode", choices=["full", "test", "preprocess", "train", "evaluate"], 
                       default="test", help="실행 모드")
    parser.add_argument("--config", default="configs/training_config.yaml", 
                       help="설정 파일 경로")
    parser.add_argument("--skip", nargs="+", type=int, default=[], 
                       help="건너뛸 단계 번호 (1-5)")
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = PillRecognitionPipeline(args.config)
    
    if args.mode == "full":
        # 전체 파이프라인 실행
        pipeline.run_full_pipeline(skip_steps=args.skip)
    
    elif args.mode == "test":
        # 빠른 테스트 실행
        pipeline.run_quick_test()
    
    elif args.mode == "preprocess":
        # 데이터 전처리만 실행
        pipeline.step1_data_preprocessing()
    
    elif args.mode == "train":
        # 학습만 실행
        dataset_yaml = "dataset/pill_dataset.yaml"
        if not os.path.exists(dataset_yaml):
            print("❌ 데이터셋이 없습니다. 먼저 전처리를 실행하세요.")
            return
        pipeline.step3_model_training(dataset_yaml)
    
    elif args.mode == "evaluate":
        # 평가만 실행
        results_dir = Path("results")
        model_paths = list(results_dir.glob("*/best_model.pt"))
        if not model_paths:
            print("❌ 학습된 모델을 찾을 수 없습니다.")
            return
        
        model_path = model_paths[-1]
        dataset_yaml = "dataset/pill_dataset.yaml"
        pipeline.step4_model_evaluation(str(model_path), dataset_yaml)


if __name__ == "__main__":
    main()

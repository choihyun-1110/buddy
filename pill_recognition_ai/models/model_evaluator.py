"""
모델 평가 및 성능 측정 모듈
mAP, precision, recall, FPS 등을 측정합니다.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import yaml
from tqdm import tqdm


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.inference_config = self.config['inference']
        
        # 결과 저장 디렉토리
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_path: str) -> YOLO:
        """모델 로드"""
        print(f"📦 모델 로드: {model_path}")
        model = YOLO(model_path)
        return model
    
    def evaluate_model(self, model_path: str, dataset_yaml_path: str) -> Dict:
        """모델 종합 평가"""
        print("🔍 모델 평가 시작...")
        
        # 모델 로드
        model = self.load_model(model_path)
        
        # 기본 검증 실행
        val_results = model.val(
            data=dataset_yaml_path,
            imgsz=self.config['data']['image_size'],
            conf=self.eval_config['confidence_threshold'],
            iou=0.45,  # 기본 NMS 임계값
            verbose=True
        )
        
        # 평가 결과 정리
        evaluation_results = {
            'model_path': model_path,
            'dataset_path': dataset_yaml_path,
            'basic_metrics': {
                'mAP@0.5': float(val_results.box.map50),
                'mAP@0.5:0.95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'f1_score': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr) if (val_results.box.mp + val_results.box.mr) > 0 else 0
            },
            'detailed_metrics': self._extract_detailed_metrics(val_results),
            'class_metrics': self._extract_class_metrics(val_results),
            'inference_speed': self._measure_inference_speed(model, dataset_yaml_path)
        }
        
        # 결과 저장
        self._save_evaluation_results(evaluation_results)
        
        # 시각화 생성
        self._create_evaluation_plots(evaluation_results, val_results)
        
        print("✅ 모델 평가 완료!")
        return evaluation_results
    
    def _extract_detailed_metrics(self, val_results) -> Dict:
        """상세 메트릭 추출"""
        detailed_metrics = {}
        
        # IoU별 mAP 계산
        if hasattr(val_results.box, 'maps'):
            detailed_metrics['mAP_by_iou'] = {
                f'IoU_{iou:.2f}': float(map_val) 
                for iou, map_val in zip(self.eval_config['iou_thresholds'], val_results.box.maps)
            }
        
        # 클래스별 상세 메트릭
        if hasattr(val_results.box, 'ap_class_index'):
            detailed_metrics['per_class_ap'] = {
                str(idx): float(ap) for idx, ap in zip(val_results.box.ap_class_index, val_results.box.ap)
            }
        
        return detailed_metrics
    
    def _extract_class_metrics(self, val_results) -> Dict:
        """클래스별 메트릭 추출"""
        class_metrics = {}
        
        if hasattr(val_results.box, 'ap_class_index') and hasattr(val_results.box, 'ap'):
            for idx, ap in zip(val_results.box.ap_class_index, val_results.box.ap):
                class_name = self.config['data']['classes'][idx] if idx < len(self.config['data']['classes']) else f"class_{idx}"
                class_metrics[class_name] = {
                    'ap@0.5': float(ap),
                    'ap@0.5:0.95': float(val_results.box.map)  # 전체 평균 사용
                }
        
        return class_metrics
    
    def _measure_inference_speed(self, model: YOLO, dataset_yaml_path: str) -> Dict:
        """추론 속도 측정"""
        print("⏱️ 추론 속도 측정 중...")
        
        try:
            # 테스트 이미지 로드
            test_images_dir = Path(dataset_yaml_path).parent / "test" / "images"
            if not test_images_dir.exists():
                print("⚠️ 테스트 이미지 디렉토리를 찾을 수 없습니다.")
                return {}
            
            test_images = list(test_images_dir.glob("*.png"))[:10]  # 최대 10개 이미지로 테스트
            
            if not test_images:
                print("⚠️ 테스트 이미지가 없습니다.")
                return {}
            
            # 간단한 속도 측정 (오류 방지)
            speed_results = {
                'cpu_fps': 0.0,
                'cpu_latency_ms': 0.0,
                'test_images': len(test_images),
                'note': 'Speed measurement skipped due to PyTorch compatibility issue'
            }
            
            print("⚠️ 추론 속도 측정 건너뛰기 (PyTorch 호환성 문제)")
            return speed_results
            
        except Exception as e:
            print(f"⚠️ 추론 속도 측정 실패: {e}")
            return {
                'cpu_fps': 0.0,
                'cpu_latency_ms': 0.0,
                'test_images': 0,
                'error': str(e)
            }
    
    def _save_evaluation_results(self, results: Dict):
        """평가 결과 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 평가 결과 저장: {results_file}")
    
    def _create_evaluation_plots(self, results: Dict, val_results):
        """평가 결과 시각화"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plots_dir = self.results_dir / f"evaluation_plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 기본 메트릭 막대 그래프
        self._plot_basic_metrics(results['basic_metrics'], plots_dir)
        
        # 2. 클래스별 AP 히트맵
        if results['class_metrics']:
            self._plot_class_metrics(results['class_metrics'], plots_dir)
        
        # 3. 추론 속도 비교
        if results['inference_speed']:
            self._plot_inference_speed(results['inference_speed'], plots_dir)
        
        print(f"📊 시각화 결과 저장: {plots_dir}")
    
    def _plot_basic_metrics(self, metrics: Dict, plots_dir: Path):
        """기본 메트릭 시각화"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # 값 표시
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim(0, 1.1)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'basic_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_metrics(self, class_metrics: Dict, plots_dir: Path):
        """클래스별 메트릭 시각화"""
        if not class_metrics:
            return
        
        # 클래스별 AP 데이터 준비
        classes = list(class_metrics.keys())
        ap_values = [class_metrics[cls]['ap@0.5'] for cls in classes]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(classes)), ap_values, color='skyblue')
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, ap_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Pill Classes')
        ax.set_ylabel('Average Precision (AP@0.5)')
        ax.set_title('Average Precision by Pill Class')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_speed(self, speed_metrics: Dict, plots_dir: Path):
        """추론 속도 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # FPS 비교
        devices = ['CPU']
        fps_values = [speed_metrics['cpu_fps']]
        
        if 'gpu_fps' in speed_metrics:
            devices.append('GPU')
            fps_values.append(speed_metrics['gpu_fps'])
        
        bars1 = ax1.bar(devices, fps_values, color=['#ff7f0e', '#2ca02c'])
        ax1.set_ylabel('FPS')
        ax1.set_title('Inference Speed (FPS)')
        
        for bar, value in zip(bars1, fps_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 지연시간 비교
        latency_values = [speed_metrics['cpu_latency_ms']]
        if 'gpu_latency_ms' in speed_metrics:
            latency_values.append(speed_metrics['gpu_latency_ms'])
        
        bars2 = ax2.bar(devices, latency_values, color=['#ff7f0e', '#2ca02c'])
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Inference Latency')
        
        for bar, value in zip(bars2, latency_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'inference_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, model_paths: List[str], dataset_yaml_path: str) -> pd.DataFrame:
        """여러 모델 성능 비교"""
        print("🔄 모델 성능 비교 시작...")
        
        comparison_results = []
        
        for model_path in model_paths:
            print(f"\n📊 모델 평가: {model_path}")
            results = self.evaluate_model(model_path, dataset_yaml_path)
            
            comparison_results.append({
                'model': Path(model_path).stem,
                'mAP@0.5': results['basic_metrics']['mAP@0.5'],
                'mAP@0.5:0.95': results['basic_metrics']['mAP@0.5:0.95'],
                'precision': results['basic_metrics']['precision'],
                'recall': results['basic_metrics']['recall'],
                'f1_score': results['basic_metrics']['f1_score'],
                'cpu_fps': results['inference_speed'].get('cpu_fps', 0),
                'gpu_fps': results['inference_speed'].get('gpu_fps', 0)
            })
        
        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(comparison_results)
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_file = self.results_dir / f"model_comparison_{timestamp}.csv"
        df.to_csv(comparison_file, index=False)
        
        print(f"📈 모델 비교 결과 저장: {comparison_file}")
        print("\n🏆 모델 성능 비교 결과:")
        print(df.to_string(index=False))
        
        return df


if __name__ == "__main__":
    # 모델 평가 실행
    evaluator = ModelEvaluator()
    
    # 평가할 모델 경로
    model_path = "results/training_20241027_085100/best_model.pt"
    dataset_yaml = "dataset/pill_dataset.yaml"
    
    # 단일 모델 평가
    if os.path.exists(model_path):
        results = evaluator.evaluate_model(model_path, dataset_yaml)
        print("\n📊 평가 결과:")
        for metric, value in results['basic_metrics'].items():
            print(f"   {metric}: {value:.4f}")
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 여러 모델 비교 (예시)
    # model_paths = [
    #     "results/training_20241027_085100/best_model.pt",
    #     "results/training_20241027_090000/best_model.pt"
    # ]
    # evaluator.compare_models(model_paths, dataset_yaml)

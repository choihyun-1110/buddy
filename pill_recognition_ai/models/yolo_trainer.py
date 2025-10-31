"""
YOLOv8 기반 알약 인식 모델 학습 파이프라인
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import wandb
from datetime import datetime
import shutil


class PillModelTrainer:
    """알약 인식 모델 학습 클래스"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.model_config = self.config['model']
        self.data_config = self.config['data']
        
        # 결과 저장 디렉토리 생성
        self.results_dir = Path("results") / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 디렉토리 생성
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_wandb(self):
        """Weights & Biases 설정"""
        if self.training_config.get('use_wandb', False):
            wandb.init(
                project="pill-recognition-ai",
                name=f"pill_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model': self.model_config,
                    'training': self.training_config,
                    'data': self.data_config
                }
            )
            print("✅ Weights & Biases 초기화 완료")
    
    def create_training_config(self, dataset_yaml_path: str) -> dict:
        """학습 설정 생성"""
        training_args = {
            # 데이터 설정
            'data': dataset_yaml_path,
            
            # 모델 설정
            'model': self.model_config['model_name'],
            'imgsz': self.data_config['image_size'],
            
            # 학습 설정
            'epochs': self.training_config['epochs'],
            'batch': self.training_config['batch_size'],
            'lr0': self.training_config['learning_rate'],
            'lrf': self.training_config['lr_final'],
            'momentum': self.training_config['momentum'],
            'weight_decay': self.training_config['weight_decay'],
            'warmup_epochs': self.training_config['warmup_epochs'],
            'warmup_momentum': self.training_config['warmup_momentum'],
            'warmup_bias_lr': self.training_config['warmup_bias_lr'],
            
            # 옵티마이저 설정
            'optimizer': self.training_config['optimizer'],
            'cos_lr': self.training_config['cosine_lr'],
            
            # 증강 설정
            'hsv_h': self.training_config['augmentation']['hsv_h'],
            'hsv_s': self.training_config['augmentation']['hsv_s'],
            'hsv_v': self.training_config['augmentation']['hsv_v'],
            'degrees': self.training_config['augmentation']['rotation'],
            'translate': self.training_config['augmentation']['translate'],
            'scale': self.training_config['augmentation']['scale'],
            'shear': self.training_config['augmentation']['shear'],
            'perspective': self.training_config['augmentation']['perspective'],
            'flipud': self.training_config['augmentation']['flipud'],
            'fliplr': self.training_config['augmentation']['fliplr'],
            'mosaic': self.training_config['augmentation']['mosaic'],
            'mixup': self.training_config['augmentation']['mixup'],
            'copy_paste': self.training_config['augmentation']['copy_paste'],
            
            # 검증 설정
            'val': True,
            'save_period': self.training_config['save_period'],
            'save': True,
            'save_txt': True,
            'save_conf': True,
            
            # 로깅 설정
            'plots': True,
            'verbose': True,
            
            # 결과 저장 경로
            'project': str(self.results_dir),
            'name': 'pill_detection',
            
            # 기타 설정
            'device': self.training_config.get('device', 'auto'),
            'workers': self.training_config.get('workers', 8),
            'patience': self.training_config.get('patience', 50),
            'freeze': self.training_config.get('freeze', None),
            'resume': self.training_config.get('resume', False),
        }
        
        return training_args
    
    def train_model(self, dataset_yaml_path: str):
        """모델 학습 실행"""
        print("🚀 알약 인식 모델 학습 시작...")
        
        # Weights & Biases 설정
        self.setup_wandb()
        
        # 모델 로드
        model_name = self.model_config['model_name']
        print(f"📦 모델 로드: {model_name}")
        model = YOLO(model_name)
        
        # 학습 설정 생성
        training_args = self.create_training_config(dataset_yaml_path)
        
        # 학습 시작
        print("🎯 학습 시작...")
        results = model.train(**training_args)
        
        # 학습 결과 저장
        self.save_training_results(results, training_args)
        
        print("✅ 모델 학습 완료!")
        return results
    
    def save_training_results(self, results, training_args):
        """학습 결과 저장"""
        # 학습 설정 저장
        config_save_path = self.results_dir / "training_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'training_args': training_args,
                'model_config': self.model_config,
                'data_config': self.data_config
            }, f, default_flow_style=False, allow_unicode=True)
        
        # 최고 모델 경로 저장
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            # 결과 디렉토리로 복사
            shutil.copy2(best_model_path, self.results_dir / "best_model.pt")
            print(f"🏆 최고 모델 저장: {self.results_dir / 'best_model.pt'}")
        
        # 학습 로그 저장
        log_files = ['results.csv', 'train_batch0.jpg', 'val_batch0_pred.jpg', 'confusion_matrix.png']
        for log_file in log_files:
            src_path = results.save_dir / log_file
            if src_path.exists():
                shutil.copy2(src_path, self.results_dir / log_file)
        
        print(f"📊 학습 결과 저장 완료: {self.results_dir}")
    
    def validate_model(self, model_path: str, dataset_yaml_path: str):
        """모델 검증"""
        print("🔍 모델 검증 시작...")
        
        # 모델 로드
        model = YOLO(model_path)
        
        # 검증 실행
        results = model.val(data=dataset_yaml_path, imgsz=self.data_config['image_size'])
        
        # 검증 결과 출력
        print("📈 검증 결과:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(self, model_path: str, export_formats: list = ['onnx', 'tflite']):
        """모델 내보내기 (ONNX, TensorFlow Lite 등)"""
        print("📤 모델 내보내기 시작...")
        
        # 모델 로드
        model = YOLO(model_path)
        
        exported_models = {}
        
        for format_name in export_formats:
            try:
                print(f"🔄 {format_name.upper()} 형식으로 내보내기...")
                
                if format_name == 'onnx':
                    exported_path = model.export(format='onnx', imgsz=self.data_config['image_size'])
                elif format_name == 'tflite':
                    exported_path = model.export(format='tflite', imgsz=self.data_config['image_size'])
                elif format_name == 'coreml':
                    exported_path = model.export(format='coreml', imgsz=self.data_config['image_size'])
                else:
                    print(f"⚠️ 지원하지 않는 형식: {format_name}")
                    continue
                
                exported_models[format_name] = exported_path
                print(f"✅ {format_name.upper()} 내보내기 완료: {exported_path}")
                
            except Exception as e:
                print(f"❌ {format_name.upper()} 내보내기 실패: {e}")
        
        return exported_models


if __name__ == "__main__":
    # 학습 실행
    trainer = PillModelTrainer()
    
    # 데이터셋 YAML 파일 경로
    dataset_yaml = "dataset/pill_dataset.yaml"
    
    # 모델 학습
    results = trainer.train_model(dataset_yaml)
    
    # 모델 검증
    best_model_path = trainer.results_dir / "best_model.pt"
    if best_model_path.exists():
        trainer.validate_model(str(best_model_path), dataset_yaml)
        
        # 모델 내보내기
        trainer.export_model(str(best_model_path), ['onnx', 'tflite'])

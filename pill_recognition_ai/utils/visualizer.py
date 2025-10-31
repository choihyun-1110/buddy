"""
데이터 전처리 및 모델 결과 시각화 모듈
바운딩 박스, 예측 결과 등을 시각화합니다.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Dict
import random
from PIL import Image, ImageDraw, ImageFont
import yaml


class ResultVisualizer:
    """결과 시각화 클래스"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['data']['classes']
        self.class_colors = self._generate_colors(len(self.classes))
        
        # 결과 저장 디렉토리
        self.output_dir = Path("results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """클래스별 색상 생성"""
        colors = []
        for i in range(num_classes):
            # HSV 색상 공간에서 균등하게 분포 (0-179 범위로 제한)
            hue = int(179 * i / num_classes)
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(rgb))
        return colors
    
    def visualize_bounding_boxes(self, image_path: str, bbox_data: List[Dict], 
                                output_path: str = None, title: str = "Bounding Box Visualization"):
        """바운딩 박스 시각화"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # matplotlib으로 시각화
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 바운딩 박스 그리기
        for bbox_info in bbox_data:
            class_id = bbox_info['class_id']
            class_name = self.classes[class_id]
            color = self.class_colors[class_id]
            
            # YOLO 형식을 절대 좌표로 변환
            center_x, center_y, bbox_w, bbox_h = bbox_info['bbox']
            x = (center_x - bbox_w / 2) * width
            y = (center_y - bbox_h / 2) * height
            w = bbox_w * width
            h = bbox_h * height
            
            # 바운딩 박스 그리기
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=[c/255 for c in color], 
                                   facecolor='none')
            ax.add_patch(rect)
            
            # 클래스 라벨 추가
            ax.text(x, y-5, f"{class_name}", fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=[c/255 for c in color], alpha=0.7),
                   color='white', weight='bold')
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')
        
        # 결과 저장
        if output_path is None:
            output_path = self.output_dir / f"bbox_visualization_{Path(image_path).stem}.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 바운딩 박스 시각화 저장: {output_path}")
        return output_path
    
    def visualize_data_preprocessing_results(self, dataset_path: str = "dataset"):
        """데이터 전처리 결과 시각화"""
        print("📊 데이터 전처리 결과 시각화 중...")
        
        dataset_path = Path(dataset_path)
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if not images_dir.exists():
                print(f"⚠️ {split} 이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
                continue
            
            # 샘플 이미지들 선택
            image_files = list(images_dir.glob('*.png'))[:5]  # 최대 5개
            
            for i, img_file in enumerate(image_files):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    continue
                
                # 라벨 데이터 읽기
                bbox_data = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            bbox_data.append({
                                'class_id': class_id,
                                'bbox': bbox
                            })
                
                # 시각화
                output_path = self.output_dir / f"{split}_sample_{i+1}.png"
                self.visualize_bounding_boxes(
                    str(img_file), 
                    bbox_data, 
                    str(output_path),
                    f"{split.upper()} Sample {i+1} - {len(bbox_data)} pills detected"
                )
        
        print(f"✅ 데이터 전처리 시각화 완료: {self.output_dir}")
    
    def visualize_augmentation_results(self, original_dataset: str = "dataset", 
                                     augmented_dataset: str = "dataset_augmented"):
        """데이터 증강 결과 시각화"""
        print("🔄 데이터 증강 결과 시각화 중...")
        
        original_path = Path(original_dataset)
        augmented_path = Path(augmented_dataset)
        
        if not augmented_path.exists():
            print(f"⚠️ 증강된 데이터셋을 찾을 수 없습니다: {augmented_path}")
            return
        
        # 원본과 증강된 이미지 비교
        splits = ['train', 'val', 'test']
        
        for split in splits:
            orig_images = list((original_path / split / 'images').glob('*.png'))[:3]
            aug_images = list((augmented_path / split / 'images').glob('*.png'))[:3]
            
            # 원본 이미지들 시각화
            for i, img_file in enumerate(orig_images):
                label_file = original_path / split / 'labels' / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    bbox_data = self._read_label_file(label_file)
                    output_path = self.output_dir / f"original_{split}_{i+1}.png"
                    self.visualize_bounding_boxes(
                        str(img_file), 
                        bbox_data, 
                        str(output_path),
                        f"Original {split.upper()} Sample {i+1}"
                    )
            
            # 증강된 이미지들 시각화
            for i, img_file in enumerate(aug_images):
                label_file = augmented_path / split / 'labels' / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    bbox_data = self._read_label_file(label_file)
                    output_path = self.output_dir / f"augmented_{split}_{i+1}.png"
                    self.visualize_bounding_boxes(
                        str(img_file), 
                        bbox_data, 
                        str(output_path),
                        f"Augmented {split.upper()} Sample {i+1}"
                    )
        
        print(f"✅ 데이터 증강 시각화 완료: {self.output_dir}")
    
    def _read_label_file(self, label_file: Path) -> List[Dict]:
        """라벨 파일 읽기"""
        bbox_data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bbox_data.append({
                        'class_id': class_id,
                        'bbox': bbox
                    })
        return bbox_data
    
    def visualize_model_predictions(self, model_path: str, test_images_dir: str, 
                                   num_samples: int = 5):
        """모델 예측 결과 시각화"""
        print("🔍 모델 예측 결과 시각화 중...")
        
        try:
            from ultralytics import YOLO
            
            # 모델 로드
            model = YOLO(model_path)
            
            # 테스트 이미지들 선택
            test_images = list(Path(test_images_dir).glob('*.png'))[:num_samples]
            
            for i, img_path in enumerate(test_images):
                # 예측 실행
                results = model(str(img_path))
                
                # 결과 시각화
                result_img = results[0].plot()
                
                # 저장
                output_path = self.output_dir / f"prediction_sample_{i+1}.png"
                cv2.imwrite(str(output_path), result_img)
                
                print(f"✅ 예측 결과 저장: {output_path}")
            
            print(f"✅ 모델 예측 시각화 완료: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ 모델 예측 시각화 실패: {e}")
    
    def create_summary_visualization(self):
        """전체 결과 요약 시각화"""
        print("📈 전체 결과 요약 시각화 중...")
        
        # 생성된 이미지들 수집
        vis_files = list(self.output_dir.glob('*.png'))
        
        if not vis_files:
            print("⚠️ 시각화 파일이 없습니다.")
            return
        
        # 그리드 형태로 요약 이미지 생성
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, img_file in enumerate(vis_files[:6]):  # 최대 6개
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(img_file.stem, fontsize=10)
            axes[i].axis('off')
        
        # 빈 축 숨기기
        for i in range(len(vis_files), 6):
            axes[i].axis('off')
        
        plt.suptitle('알약 인식 AI 모델 개발 결과 시각화', fontsize=16, weight='bold')
        plt.tight_layout()
        
        summary_path = self.output_dir / "summary_visualization.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 요약 시각화 저장: {summary_path}")
        return summary_path
    
    def visualize_class_distribution(self, dataset_path: str = "dataset"):
        """클래스 분포 시각화"""
        print("📊 클래스 분포 시각화 중...")
        
        dataset_path = Path(dataset_path)
        class_counts = {cls: 0 for cls in self.classes}
        
        # 모든 라벨 파일에서 클래스 카운트
        for split in ['train', 'val', 'test']:
            labels_dir = dataset_path / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id < len(self.classes):
                                    class_counts[self.classes[class_id]] += 1
        
        # 막대 그래프 생성
        fig, ax = plt.subplots(figsize=(15, 8))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self.class_colors[i] for i in range(len(classes))]
        
        bars = ax.bar(range(len(classes)), counts, color=[c/255 for c in colors])
        
        # 값 표시
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(count), ha='center', va='bottom')
        
        ax.set_xlabel('Pill Classes', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Class Distribution in Dataset', fontsize=14, weight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        plt.tight_layout()
        
        dist_path = self.output_dir / "class_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 클래스 분포 시각화 저장: {dist_path}")
        return dist_path


if __name__ == "__main__":
    # 시각화 실행
    visualizer = ResultVisualizer()
    
    # 데이터 전처리 결과 시각화
    visualizer.visualize_data_preprocessing_results()
    
    # 클래스 분포 시각화
    visualizer.visualize_class_distribution()
    
    # 요약 시각화
    visualizer.create_summary_visualization()

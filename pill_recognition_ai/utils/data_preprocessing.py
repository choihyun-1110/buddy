"""
알약 이미지 데이터 전처리 모듈
PNG 이미지를 COCO 형식으로 변환하고 데이터셋을 구성합니다.
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm


class PillDataProcessor:
    """알약 데이터 전처리 클래스"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.source_path = Path(self.data_config['source_path'])
        self.processed_path = Path(self.data_config['processed_path'])
        self.classes = self.data_config['classes']
        self.image_size = self.data_config['image_size']
        
        # 클래스 ID 매핑 생성
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        
    def create_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        dirs_to_create = [
            self.processed_path / "train" / "images",
            self.processed_path / "train" / "labels", 
            self.processed_path / "val" / "images",
            self.processed_path / "val" / "labels",
            self.processed_path / "test" / "images", 
            self.processed_path / "test" / "labels",
            self.processed_path / "annotations"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ 디렉토리 구조 생성 완료: {self.processed_path}")
    
    def extract_pill_bbox(self, image_path: str) -> Tuple[int, int, int, int]:
        """
        알약 이미지에서 바운딩 박스 추출
        단일 알약이 포함된 이미지에서 알약의 위치를 찾습니다.
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 임계값 처리 (Otsu 방법)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 컨투어를 찾지 못한 경우 전체 이미지 사용
            h, w = image.shape[:2]
            return 0, 0, w, h
        
        # 가장 큰 컨투어 선택 (알약으로 가정)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 약간의 여백 추가 (10%)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)
        
        return x, y, w, h
    
    def convert_to_yolo_format(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        바운딩 박스를 YOLO 형식으로 변환 (정규화된 중심점 좌표와 크기)
        """
        x, y, w, h = bbox
        
        # 중심점 계산
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 정규화
        norm_center_x = center_x / img_width
        norm_center_y = center_y / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        return norm_center_x, norm_center_y, norm_width, norm_height
    
    def process_single_class(self, class_name: str) -> List[Dict]:
        """단일 클래스의 모든 이미지 처리"""
        class_path = self.source_path / class_name
        if not class_path.exists():
            print(f"⚠️ 클래스 폴더를 찾을 수 없습니다: {class_path}")
            return []
        
        class_id = self.class_to_id[class_name]
        processed_images = []
        
        # PNG 파일 목록 가져오기
        image_files = list(class_path.glob("*.png"))
        print(f"📁 클래스 {class_name}: {len(image_files)}개 이미지 처리 중...")
        
        for img_path in tqdm(image_files, desc=f"처리 중 - {class_name}"):
            try:
                # 이미지 로드하여 크기 확인
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # 바운딩 박스 추출
                bbox = self.extract_pill_bbox(img_path)
                
                # YOLO 형식으로 변환
                yolo_bbox = self.convert_to_yolo_format(bbox, img_width, img_height)
                
                processed_images.append({
                    'image_path': img_path,
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': bbox,
                    'yolo_bbox': yolo_bbox,
                    'img_width': img_width,
                    'img_height': img_height
                })
                
            except Exception as e:
                print(f"❌ 이미지 처리 실패 {img_path}: {e}")
                continue
        
        return processed_images
    
    def split_dataset(self, all_images: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """데이터셋을 train/val/test로 분할"""
        random.shuffle(all_images)
        
        total_count = len(all_images)
        train_count = int(total_count * self.data_config['split_ratio']['train'])
        val_count = int(total_count * self.data_config['split_ratio']['val'])
        
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]
        
        print(f"📊 데이터셋 분할 완료:")
        print(f"   Train: {len(train_images)}개")
        print(f"   Val: {len(val_images)}개") 
        print(f"   Test: {len(test_images)}개")
        
        return train_images, val_images, test_images
    
    def save_dataset_split(self, images: List[Dict], split_name: str):
        """데이터셋 분할 저장"""
        split_path = self.processed_path / split_name
        
        for idx, img_data in enumerate(tqdm(images, desc=f"{split_name} 저장 중")):
            # 이미지 파일명 생성
            img_filename = f"{img_data['class_name']}_{idx:04d}.png"
            img_dest_path = split_path / "images" / img_filename
            
            # 이미지 복사
            shutil.copy2(img_data['image_path'], img_dest_path)
            
            # 라벨 파일 생성 (YOLO 형식)
            label_filename = f"{img_data['class_name']}_{idx:04d}.txt"
            label_dest_path = split_path / "labels" / label_filename
            
            with open(label_dest_path, 'w') as f:
                center_x, center_y, width, height = img_data['yolo_bbox']
                f.write(f"{img_data['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def create_yolo_config(self):
        """YOLO 학습용 설정 파일 생성"""
        yolo_config = {
            'path': str(self.processed_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        config_path = self.processed_path / "pill_dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ YOLO 설정 파일 생성: {config_path}")
        return config_path
    
    def process_all_data(self):
        """전체 데이터 처리 파이프라인 실행"""
        print("🚀 알약 데이터 전처리 시작...")
        
        # 디렉토리 구조 생성
        self.create_directory_structure()
        
        # 모든 클래스 처리
        all_images = []
        for class_name in self.classes:
            class_images = self.process_single_class(class_name)
            all_images.extend(class_images)
        
        print(f"📈 총 {len(all_images)}개 이미지 처리 완료")
        
        # 데이터셋 분할
        train_images, val_images, test_images = self.split_dataset(all_images)
        
        # 분할별 저장
        self.save_dataset_split(train_images, "train")
        self.save_dataset_split(val_images, "val")
        self.save_dataset_split(test_images, "test")
        
        # YOLO 설정 파일 생성
        yolo_config_path = self.create_yolo_config()
        
        print("✅ 데이터 전처리 완료!")
        print(f"📁 처리된 데이터 경로: {self.processed_path}")
        print(f"⚙️ YOLO 설정 파일: {yolo_config_path}")
        
        return yolo_config_path


if __name__ == "__main__":
    # 데이터 처리 실행
    processor = PillDataProcessor()
    processor.process_all_data()

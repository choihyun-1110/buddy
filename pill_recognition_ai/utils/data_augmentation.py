"""
알약 이미지 데이터 증강 모듈
여러 알약을 하나의 이미지에 합성하여 다중 객체 탐지 학습용 데이터를 생성합니다.
"""

import os
import random
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from tqdm import tqdm


class PillDataAugmenter:
    """알약 데이터 증강 클래스"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['data']['augmentation']
        self.image_size = self.config['data']['image_size']
        
        # 배경 텍스처 생성
        self.background_textures = self._generate_background_textures()
        
        # 기본 증강 파이프라인 설정
        self.basic_augmentation = A.Compose([
            A.Rotate(limit=self.aug_config['basic_augmentation']['rotation'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_config['basic_augmentation']['brightness'],
                contrast_limit=self.aug_config['basic_augmentation']['contrast'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(self.aug_config['basic_augmentation']['hue'] * 180),
                sat_shift_limit=int(self.aug_config['basic_augmentation']['saturation'] * 100),
                val_shift_limit=int(self.aug_config['basic_augmentation']['brightness'] * 100),
                p=0.5
            ),
        ])
    
    def _generate_background_textures(self) -> List[np.ndarray]:
        """다양한 배경 텍스처 생성"""
        textures = []
        
        # 단색 배경들
        colors = [
            (255, 255, 255),  # 흰색
            (240, 240, 240),  # 연한 회색
            (200, 200, 200),  # 회색
            (180, 180, 180),  # 진한 회색
            (220, 220, 255),  # 연한 파란색
            (255, 220, 220),  # 연한 빨간색
            (220, 255, 220),  # 연한 초록색
        ]
        
        for color in colors:
            texture = np.full((self.image_size, self.image_size, 3), color, dtype=np.uint8)
            textures.append(texture)
        
        # 노이즈가 있는 배경들
        for _ in range(3):
            texture = np.random.randint(200, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            textures.append(texture)
        
        return textures
    
    def _extract_pill_from_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """이미지에서 알약 영역 추출 (정확한 segmentation)"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거 및 임계값 처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 컨투어를 찾지 못한 경우 전체 이미지 사용
            return image, (0, 0, image.shape[1], image.shape[0])
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 더 정확한 segmentation mask 생성
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # 모폴로지 연산으로 마스크 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 알약만 추출 (배경 완전 제거)
        pill_only = cv2.bitwise_and(image, image, mask=mask)
        
        # 투명 배경으로 변환 (더 정확한 마스크 사용)
        pill_with_alpha = self._create_transparent_pill(pill_only, mask)
        
        # 바운딩 박스 계산 (여백 포함)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # 최종 알약 영역 추출
        pill_region = pill_with_alpha[y:y+h, x:x+w]
        
        return pill_region, (x, y, w, h)
    
    def _create_transparent_pill(self, pill_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """투명 배경을 가진 알약 이미지 생성"""
        # RGBA 채널로 변환
        if len(pill_image.shape) == 3:
            h, w, c = pill_image.shape
            pill_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            pill_rgba[:, :, :3] = pill_image
            pill_rgba[:, :, 3] = mask  # 알파 채널로 마스크 사용
        else:
            pill_rgba = pill_image
        
        return pill_rgba
    
    def _blend_pill_on_background(self, background: np.ndarray, pill_image: np.ndarray, x: int, y: int):
        """투명 배경 알약을 배경에 블렌딩"""
        pill_h, pill_w = pill_image.shape[:2]
        
        # 알약이 RGBA 채널을 가지고 있는 경우
        if len(pill_image.shape) == 3 and pill_image.shape[2] == 4:
            # 알파 채널 추출
            alpha = pill_image[:, :, 3] / 255.0
            
            # RGB 채널 추출
            pill_rgb = pill_image[:, :, :3]
            
            # 배경 영역 추출
            bg_region = background[y:y+pill_h, x:x+pill_w]
            
            # 알파 블렌딩
            for c in range(3):
                background[y:y+pill_h, x:x+pill_w, c] = (
                    alpha * pill_rgb[:, :, c] + (1 - alpha) * bg_region[:, :, c]
                )
        else:
            # 일반 이미지인 경우 직접 복사
            background[y:y+pill_h, x:x+pill_w] = pill_image
    
    def _resize_pill(self, pill_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """알약 이미지 크기 조정 (비율 유지, RGBA 지원)"""
        h, w = pill_image.shape[:2]
        target_w, target_h = target_size
        
        # 비율 계산
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 리사이즈
        resized = cv2.resize(pill_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 채널 수에 따라 패딩 생성
        if len(pill_image.shape) == 3 and pill_image.shape[2] == 4:
            # RGBA 채널
            padded = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        else:
            # RGB 채널
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 중앙에 배치
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return padded
    
    def _check_overlap(self, new_bbox: Tuple[int, int, int, int], existing_bboxes: List[Tuple[int, int, int, int]], 
                      threshold: float = 0.1) -> bool:
        """바운딩 박스 겹침 확인"""
        x1, y1, w1, h1 = new_bbox
        
        for x2, y2, w2, h2 in existing_bboxes:
            # 겹치는 영역 계산
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            # 각 박스의 면적
            area1 = w1 * h1
            area2 = w2 * h2
            
            # IoU 계산
            union_area = area1 + area2 - overlap_area
            iou = overlap_area / union_area if union_area > 0 else 0
            
            if iou > threshold:
                return True
        
        return False
    
    def create_multi_pill_image(self, pill_images: List[np.ndarray], pill_classes: List[int], 
                               max_attempts: int = 100) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]:
        """여러 알약이 포함된 이미지 생성"""
        # 배경 선택
        background = random.choice(self.background_textures).copy()
        
        # 알약 개수 결정
        num_pills = random.randint(
            self.aug_config['multi_pill_synthesis']['min_pills_per_image'],
            self.aug_config['multi_pill_synthesis']['max_pills_per_image']
        )
        
        # 실제 사용할 알약 선택
        selected_indices = random.sample(range(len(pill_images)), min(num_pills, len(pill_images)))
        selected_pills = [pill_images[i] for i in selected_indices]
        selected_classes = [pill_classes[i] for i in selected_indices]
        
        placed_bboxes = []
        annotations = []
        
        for pill_img, class_id in zip(selected_pills, selected_classes):
            # 알약 크기 랜덤 조정 (원본의 0.5~1.5배)
            scale_factor = random.uniform(0.5, 1.5)
            target_size = (int(self.image_size * 0.2 * scale_factor), int(self.image_size * 0.2 * scale_factor))
            
            # 알약 크기 조정
            resized_pill = self._resize_pill(pill_img, target_size)
            pill_h, pill_w = resized_pill.shape[:2]
            
            # 배치 위치 찾기
            placed = False
            for attempt in range(max_attempts):
                # 랜덤 위치 생성
                x = random.randint(0, self.image_size - pill_w)
                y = random.randint(0, self.image_size - pill_h)
                
                new_bbox = (x, y, pill_w, pill_h)
                
                # 겹침 확인
                if not self._check_overlap(new_bbox, placed_bboxes):
                    # 투명 배경 알약을 배경에 배치
                    self._blend_pill_on_background(background, resized_pill, x, y)
                    
                    # 어노테이션 추가 (YOLO 형식)
                    center_x = (x + pill_w / 2) / self.image_size
                    center_y = (y + pill_h / 2) / self.image_size
                    norm_w = pill_w / self.image_size
                    norm_h = pill_h / self.image_size
                    
                    annotations.append((class_id, center_x, center_y, norm_w, norm_h))
                    placed_bboxes.append(new_bbox)
                    placed = True
                    break
            
            if not placed:
                print(f"⚠️ 알약 배치 실패 (시도 횟수 초과)")
        
        return background, annotations
    
    def augment_dataset(self, dataset_path: str, output_path: str, augmentation_factor: int = 3):
        """데이터셋 증강 실행"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        # 출력 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 각 분할에 대해 증강 수행
        for split in ['train', 'val', 'test']:
            print(f"[{split}] 데이터 증강 중...")
            
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if not images_dir.exists():
                print(f"⚠️ {split} 이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
                continue
            
            # 모든 이미지 파일 수집
            image_files = list(images_dir.glob('*.png'))
            print(f"[{split}] {len(image_files)}개 원본 이미지")
            
            # 원본 데이터 복사
            for img_file in tqdm(image_files, desc=f"{split} 원본 복사"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # 이미지 복사
                    shutil.copy2(img_file, output_path / split / 'images' / img_file.name)
                    # 라벨 복사
                    shutil.copy2(label_file, output_path / split / 'labels' / f"{img_file.stem}.txt")
            
            # 증강 데이터 생성
            augmented_count = 0
            for _ in range(augmentation_factor):
                for img_file in tqdm(image_files, desc=f"{split} 증강 생성"):
                    try:
                        # 원본 이미지에서 알약 추출
                        pill_img, _ = self._extract_pill_from_image(img_file)
                        
                        # 라벨 파일에서 클래스 정보 읽기
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if not label_file.exists():
                            continue
                        
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        if not lines:
                            continue
                        
                        # 클래스 정보 추출
                        pill_classes = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                pill_classes.append(class_id)
                        
                        if not pill_classes:
                            continue
                        
                        # 다른 알약 이미지 수집
                        other_images = []
                        other_classes = []
                        
                        # 같은 분할의 다른 이미지들에서 추가 알약 수집
                        for other_img_file in random.sample(image_files, min(3, len(image_files))):
                            if other_img_file != img_file:
                                try:
                                    other_pill_img, _ = self._extract_pill_from_image(other_img_file)
                                    other_label_file = labels_dir / f"{other_img_file.stem}.txt"
                                    if other_label_file.exists():
                                        with open(other_label_file, 'r') as f:
                                            label_lines = [line.strip().split() for line in f if line.strip()]

                                        if label_lines:
                                            # 여러 바운딩 박스가 존재할 경우 우선 첫 번째 클래스를 사용
                                            original_class_id = int(label_lines[0][0])
                                            other_images.append(other_pill_img)
                                            other_classes.append(original_class_id)
                                except:
                                    continue
                        
                        # 다중 알약 이미지 생성
                        all_pills = [pill_img] + other_images[:2]  # 최대 3개 알약
                        all_classes = pill_classes + other_classes[:2]
                        
                        multi_pill_img, annotations = self.create_multi_pill_image(
                            all_pills, all_classes
                        )
                        
                        if annotations:
                            # 증강된 이미지 저장
                            aug_img_name = f"{img_file.stem}_aug_{augmented_count:04d}.png"
                            aug_img_path = output_path / split / 'images' / aug_img_name
                            cv2.imwrite(str(aug_img_path), multi_pill_img)
                            
                            # 증강된 라벨 저장
                            aug_label_name = f"{img_file.stem}_aug_{augmented_count:04d}.txt"
                            aug_label_path = output_path / split / 'labels' / aug_label_name
                            
                            with open(aug_label_path, 'w') as f:
                                for ann in annotations:
                                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                            
                            augmented_count += 1
                    
                    except Exception as e:
                        print(f"❌ 증강 실패 {img_file}: {e}")
                        continue
            
            print(f"[{split}] 증강 완료: {augmented_count}개 추가 생성")


if __name__ == "__main__":
    # 데이터 증강 실행
    augmenter = PillDataAugmenter()
    
    # 원본 데이터셋 경로와 출력 경로 설정
    original_dataset = "dataset"
    augmented_dataset = "dataset_augmented"
    
    augmenter.augment_dataset(original_dataset, augmented_dataset, augmentation_factor=2)

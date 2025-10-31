#!/usr/bin/env python3
"""
개선된 segmentation으로 증강된 데이터로 기존 모델 테스트
"""

from ultralytics import YOLO
import os
import glob
import cv2
import yaml
from pathlib import Path

def main():
    # 모델 로드
    model_path = 'results/training_20251027_212659/best_model.pt'
    print(f"🔍 모델 로드: {model_path}")
    model = YOLO(model_path)

    # 개선된 증강 데이터로 테스트
    augmented_test_images_dir = 'dataset_augmented_v2/test/images'
    augmented_test_images = glob.glob(os.path.join(augmented_test_images_dir, '*.png'))[:10]  # 처음 10개만

    print(f'🎨 개선된 증강 데이터 {len(augmented_test_images)}개 테스트 이미지로 시각화 시작...')

    # 결과 저장 디렉토리 생성
    output_viz_dir = 'results/improved_augmentation_visualizations'
    os.makedirs(output_viz_dir, exist_ok=True)

    # 데이터셋 YAML 파일 로드 (클래스 이름 확인용)
    dataset_yaml_path = 'dataset_augmented_v2/pill_dataset_augmented_v2.yaml'
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_names = dataset_config.get('names', [])

    for i, img_path in enumerate(augmented_test_images):
        img_name = os.path.basename(img_path)
        
        # 추론 실행
        print(f"📸 추론 중: {img_name}")
        results = model(img_path)
        
        # 결과 시각화 (YOLO 내장 시각화 사용)
        result_img = results[0].plot()
        
        # 저장
        output_path = os.path.join(output_viz_dir, f'improved_aug_visualization_{i+1}_{img_name}')
        cv2.imwrite(output_path, result_img)
        print(f'✅ {img_name} 시각화 완료: {output_path}')

        # 라벨 파일 확인 (디버깅용)
        label_file_path = Path(augmented_test_images_dir).parent / 'labels' / f"{Path(img_path).stem}.txt"
        if label_file_path.exists():
            print(f"   ➡️ 실제 라벨 확인: {label_file_path}")
            with open(label_file_path, 'r') as f:
                labels = f.readlines()
            print(f"   ➡️ 실제 라벨 수: {len(labels)}")
            for idx, label_line in enumerate(labels):
                parts = label_line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    print(f"     라벨 {idx+1}: 클래스 {class_id} ({class_names[class_id] if class_id < len(class_names) else 'Unknown'})")
        else:
            print(f"   ➡️ 라벨 파일 없음: {label_file_path}")

        # 예측 결과 확인
        result = results[0]
        if result.boxes is not None:
            print(f"   ➡️ 예측된 객체 수: {len(result.boxes)}")
            for idx, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                print(f"     예측 {idx+1}: 클래스 {cls} ({class_names[cls] if cls < len(class_names) else 'Unknown'}), 신뢰도 {conf:.3f}")
        else:
            print(f"   ➡️ 예측된 객체 없음")

    print('🎨 개선된 증강 데이터 시각화 완료!')

if __name__ == "__main__":
    main()

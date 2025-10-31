#!/usr/bin/env python3
"""
테스트 이미지로 학습된 모델 시각화
"""

from ultralytics import YOLO
import os
import glob
import cv2

def main():
    # 모델 로드
    model_path = 'results/training_20251027_212659/best_model.pt'
    print(f"🔍 모델 로드: {model_path}")
    model = YOLO(model_path)

    # 테스트 이미지들로 추론 및 시각화
    test_images_dir = 'dataset/test/images'
    test_images = glob.glob(os.path.join(test_images_dir, '*.png'))[:5]  # 처음 5개만

    print(f'🎨 {len(test_images)}개 테스트 이미지로 시각화 시작...')

    # 결과 저장 디렉토리 생성
    os.makedirs('results/visualizations', exist_ok=True)

    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        
        # 추론 실행
        print(f"📸 추론 중: {img_name}")
        results = model(img_path)
        
        # 결과 시각화 (YOLO 내장 시각화 사용)
        result_img = results[0].plot()
        
        # 저장
        output_path = f'results/visualizations/visualization_{i+1}_{img_name}'
        cv2.imwrite(output_path, result_img)
        print(f'✅ {img_name} 시각화 완료: {output_path}')

    print('🎨 모든 시각화 완료!')

if __name__ == "__main__":
    main()

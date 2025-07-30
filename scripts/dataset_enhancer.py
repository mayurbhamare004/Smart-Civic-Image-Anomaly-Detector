#!/usr/bin/env python3
"""
Dataset Enhancement Script for Civic Anomaly Detection
Applies advanced augmentation techniques to increase dataset size and diversity
"""

import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
from collections import defaultdict
import albumentations as A
from tqdm import tqdm

class DatasetEnhancer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "scripts" / "data" / "processed"
        self.enhanced_dir = self.base_dir / "scripts" / "data" / "enhanced"
        
        # Create enhanced dataset directory
        for split in ['train', 'val', 'test']:
            (self.enhanced_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.enhanced_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Class information for targeted augmentation
        self.class_names = {
            0: "pothole",
            1: "garbage_dump", 
            2: "waterlogging",
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
        
        # Augmentation multipliers per class (to balance dataset)
        self.augmentation_multipliers = {
            0: 3,  # Potholes - increase significantly
            1: 2,  # Garbage dumps
            2: 4,  # Waterlogging - often underrepresented
            3: 2,  # Streetlights
            4: 2,  # Sidewalks
            5: 2   # Construction debris
        }
    
    def analyze_dataset_distribution(self):
        """Analyze current dataset class distribution"""
        print("📊 Analyzing dataset distribution...")
        
        class_counts = defaultdict(int)
        total_images = 0
        
        for split in ['train', 'val']:
            labels_dir = self.data_dir / split / 'labels'
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                total_images += 1
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
        
        print(f"📈 Dataset Analysis:")
        print(f"   Total images: {total_images}")
        print(f"   Total objects: {sum(class_counts.values())}")
        
        for class_id, count in class_counts.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            percentage = (count / sum(class_counts.values())) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return class_counts, total_images
    
    def create_augmentation_pipeline(self, augmentation_level='medium'):
        """Create Albumentations augmentation pipeline"""
        
        if augmentation_level == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        elif augmentation_level == 'medium':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                    A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True),
                ], p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
                A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        else:  # heavy
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40, p=0.8),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0)),
                    A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8)),
                    A.MultiplicativeNoise(multiplier=[0.8, 1.2], per_channel=True),
                ], p=0.7),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=20, p=0.8),
                A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.6),
                A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.4),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transform
    
    def load_yolo_annotations(self, label_path):
        """Load YOLO format annotations"""
        annotations = []
        if not label_path.exists():
            return annotations
            
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    annotations.append([x_center, y_center, width, height, class_id])
        
        return annotations
    
    def save_yolo_annotations(self, annotations, label_path):
        """Save YOLO format annotations"""
        with open(label_path, 'w') as f:
            for ann in annotations:
                x_center, y_center, width, height, class_id = ann
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_image_with_annotations(self, image_path, label_path, transform, num_augmentations=1):
        """Augment image with bounding box annotations"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        annotations = self.load_yolo_annotations(label_path)
        if not annotations:
            return []
        
        augmented_data = []
        
        for i in range(num_augmentations):
            try:
                # Prepare bboxes and labels for Albumentations
                bboxes = [[ann[0], ann[1], ann[2], ann[3]] for ann in annotations]
                class_labels = [ann[4] for ann in annotations]
                
                # Apply augmentation
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Check if augmentation was successful
                if len(augmented['bboxes']) > 0:
                    aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    aug_annotations = []
                    
                    for bbox, class_label in zip(augmented['bboxes'], augmented['class_labels']):
                        aug_annotations.append([bbox[0], bbox[1], bbox[2], bbox[3], class_label])
                    
                    augmented_data.append((aug_image, aug_annotations))
                
            except Exception as e:
                print(f"⚠️  Augmentation failed for {image_path}: {e}")
                continue
        
        return augmented_data
    
    def enhance_dataset_split(self, split='train', augmentation_level='medium'):
        """Enhance a specific dataset split"""
        print(f"🔄 Enhancing {split} split with {augmentation_level} augmentation...")
        
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ {split} split not found!")
            return
        
        # Copy original data first
        print("📋 Copying original data...")
        for img_file in images_dir.glob('*.jpg'):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Copy image
            target_img = self.enhanced_dir / split / 'images' / img_file.name
            cv2.imwrite(str(target_img), cv2.imread(str(img_file)))
            
            # Copy label
            if label_file.exists():
                target_label = self.enhanced_dir / split / 'labels' / label_file.name
                with open(label_file, 'r') as src, open(target_label, 'w') as dst:
                    dst.write(src.read())
        
        # Create augmentation pipeline
        transform = self.create_augmentation_pipeline(augmentation_level)
        
        # Get list of images to augment
        image_files = list(images_dir.glob('*.jpg'))
        
        # Analyze class distribution for targeted augmentation
        class_image_map = defaultdict(list)
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            annotations = self.load_yolo_annotations(label_file)
            
            for ann in annotations:
                class_id = int(ann[4])
                class_image_map[class_id].append(img_file)
        
        print(f"🎯 Applying targeted augmentation...")
        total_augmented = 0
        
        # Apply class-specific augmentation
        for class_id, multiplier in self.augmentation_multipliers.items():
            if class_id not in class_image_map:
                continue
            
            class_images = class_image_map[class_id]
            class_name = self.class_names[class_id]
            
            print(f"   Augmenting {class_name}: {len(class_images)} images × {multiplier}")
            
            for img_file in tqdm(class_images, desc=f"Augmenting {class_name}"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                # Generate augmentations
                augmented_data = self.augment_image_with_annotations(
                    img_file, label_file, transform, multiplier
                )
                
                # Save augmented data
                for j, (aug_image, aug_annotations) in enumerate(augmented_data):
                    aug_name = f"{img_file.stem}_aug_{class_id}_{j}"
                    
                    # Save augmented image
                    aug_img_path = self.enhanced_dir / split / 'images' / f"{aug_name}.jpg"
                    cv2.imwrite(str(aug_img_path), aug_image)
                    
                    # Save augmented annotations
                    aug_label_path = self.enhanced_dir / split / 'labels' / f"{aug_name}.txt"
                    self.save_yolo_annotations(aug_annotations, aug_label_path)
                    
                    total_augmented += 1
        
        print(f"✅ Enhanced {split} split: +{total_augmented} augmented images")
        return total_augmented
    
    def create_enhanced_dataset_yaml(self):
        """Create dataset.yaml for enhanced dataset"""
        dataset_config = {
            'path': str(self.enhanced_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 6,
            'names': list(self.class_names.values())
        }
        
        yaml_path = self.enhanced_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(f"path: {dataset_config['path']}\n")
            f.write(f"train: {dataset_config['train']}\n")
            f.write(f"val: {dataset_config['val']}\n")
            f.write(f"test: {dataset_config['test']}\n")
            f.write(f"nc: {dataset_config['nc']}\n")
            f.write(f"names: {dataset_config['names']}\n")
        
        print(f"📄 Enhanced dataset config saved: {yaml_path}")
        return yaml_path
    
    def generate_enhancement_report(self, original_counts, enhanced_counts):
        """Generate dataset enhancement report"""
        report = {
            'enhancement_timestamp': str(pd.Timestamp.now()),
            'original_dataset': {
                'total_images': sum(original_counts.values()),
                'class_distribution': dict(original_counts)
            },
            'enhanced_dataset': {
                'total_images': sum(enhanced_counts.values()),
                'class_distribution': dict(enhanced_counts)
            },
            'enhancement_summary': {
                'total_images_added': sum(enhanced_counts.values()) - sum(original_counts.values()),
                'enhancement_ratio': sum(enhanced_counts.values()) / sum(original_counts.values()) if sum(original_counts.values()) > 0 else 0
            }
        }
        
        report_path = self.enhanced_dir / "enhancement_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Enhancement report saved: {report_path}")
        return report_path
    
    def run_dataset_enhancement(self, augmentation_level='medium'):
        """Run complete dataset enhancement pipeline"""
        print("🚀 Dataset Enhancement Pipeline")
        print("="*50)
        
        # Analyze original dataset
        original_counts, original_images = self.analyze_dataset_distribution()
        
        if original_images == 0:
            print("❌ No original dataset found!")
            return False
        
        # Enhance training split
        train_augmented = self.enhance_dataset_split('train', augmentation_level)
        
        # Enhance validation split (lighter augmentation)
        val_augmented = self.enhance_dataset_split('val', 'light')
        
        # Copy test split without augmentation
        test_images_dir = self.data_dir / 'test' / 'images'
        test_labels_dir = self.data_dir / 'test' / 'labels'
        
        if test_images_dir.exists():
            print("📋 Copying test split...")
            for img_file in test_images_dir.glob('*.jpg'):
                label_file = test_labels_dir / f"{img_file.stem}.txt"
                
                # Copy image
                target_img = self.enhanced_dir / 'test' / 'images' / img_file.name
                cv2.imwrite(str(target_img), cv2.imread(str(img_file)))
                
                # Copy label if exists
                if label_file.exists():
                    target_label = self.enhanced_dir / 'test' / 'labels' / label_file.name
                    with open(label_file, 'r') as src, open(target_label, 'w') as dst:
                        dst.write(src.read())
        
        # Create enhanced dataset config
        self.create_enhanced_dataset_yaml()
        
        # Analyze enhanced dataset
        enhanced_counts, enhanced_images = self.analyze_enhanced_dataset()
        
        # Generate report
        self.generate_enhancement_report(original_counts, enhanced_counts)
        
        print(f"\n🎉 Dataset Enhancement Complete!")
        print(f"📈 Original: {original_images} images")
        print(f"📈 Enhanced: {enhanced_images} images")
        print(f"📈 Improvement: {enhanced_images/original_images:.1f}x")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Train with enhanced dataset: python scripts/enhanced_model_trainer.py")
        print(f"2. Update config to use enhanced dataset path")
        
        return True
    
    def analyze_enhanced_dataset(self):
        """Analyze enhanced dataset distribution"""
        class_counts = defaultdict(int)
        total_images = 0
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.enhanced_dir / split / 'labels'
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                total_images += 1
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
        
        print(f"\n📊 Enhanced Dataset Analysis:")
        print(f"   Total images: {total_images}")
        print(f"   Total objects: {sum(class_counts.values())}")
        
        for class_id, count in class_counts.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            percentage = (count / sum(class_counts.values())) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return class_counts, total_images

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Enhancement for Civic Anomaly Detection')
    parser.add_argument('--level', choices=['light', 'medium', 'heavy'], default='medium',
                       help='Augmentation level')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Only analyze dataset distribution')
    
    args = parser.parse_args()
    
    enhancer = DatasetEnhancer()
    
    if args.analyze_only:
        enhancer.analyze_dataset_distribution()
    else:
        success = enhancer.run_dataset_enhancement(args.level)
        return 0 if success else 1

if __name__ == "__main__":
    # Install required package if not available
    try:
        import albumentations as A
        import pandas as pd
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        os.system("pip install albumentations pandas tqdm")
        import albumentations as A
        import pandas as pd
        from tqdm import tqdm
    
    exit(main())
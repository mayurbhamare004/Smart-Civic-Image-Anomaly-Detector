#!/usr/bin/env python3
"""
Quick Pothole Detection Trainer
Add sample.jpg to training data and retrain model
"""

import os
import shutil
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

class QuickPotholeTrainer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.sample_image = "sample.jpg"
        self.quick_data_dir = self.base_dir / "scripts" / "data" / "quick_pothole"
        self.models_dir = self.base_dir / "models" / "weights"
        
        # Create quick training directory
        self.quick_data_dir.mkdir(parents=True, exist_ok=True)
        (self.quick_data_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.quick_data_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.quick_data_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.quick_data_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    def create_manual_annotation(self):
        """Create manual annotation for sample.jpg"""
        print("🖼️  Creating manual annotation for sample.jpg...")
        
        # Load image to get dimensions
        img = cv2.imread(self.sample_image)
        if img is None:
            print(f"❌ Could not load {self.sample_image}")
            return False
        
        height, width = img.shape[:2]
        print(f"📏 Image dimensions: {width}x{height}")
        
        # Manual annotation for the central pothole in sample.jpg
        # Image dimensions: 612x406 (width x height)
        # The main pothole appears to be in the center-right area of the image
        # These are normalized coordinates (x_center, y_center, width, height)
        annotations = [
            # Format: class_id x_center y_center width height (all normalized 0-1)
            "0 0.55 0.65 0.25 0.20",  # Main central pothole - adjusted for actual location
            "0 0.35 0.75 0.15 0.12",  # Secondary pothole area if visible
        ]
        
        # Copy sample image to training data
        train_img_path = self.quick_data_dir / "train" / "images" / "sample.jpg"
        shutil.copy2(self.sample_image, train_img_path)
        
        # Create label file
        train_label_path = self.quick_data_dir / "train" / "labels" / "sample.txt"
        with open(train_label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        # Also add to validation (duplicate for small dataset)
        val_img_path = self.quick_data_dir / "val" / "images" / "sample_val.jpg"
        val_label_path = self.quick_data_dir / "val" / "labels" / "sample_val.txt"
        shutil.copy2(self.sample_image, val_img_path)
        shutil.copy2(train_label_path, val_label_path)
        
        print("✅ Manual annotation created")
        return True
    
    def copy_existing_data(self):
        """Copy some existing training data to supplement"""
        print("📂 Copying existing pothole data...")
        
        existing_data_dir = self.base_dir / "scripts" / "data" / "processed"
        
        if not existing_data_dir.exists():
            print("⚠️  No existing processed data found")
            return
        
        # Copy some existing pothole images
        existing_train_imgs = existing_data_dir / "train" / "images"
        existing_train_labels = existing_data_dir / "train" / "labels"
        
        if existing_train_imgs.exists():
            count = 0
            for img_file in existing_train_imgs.glob("*.jpg"):
                label_file = existing_train_labels / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # Check if label contains pothole (class 0)
                    with open(label_file, 'r') as f:
                        content = f.read()
                        if content.startswith('0 ') or ' 0 ' in content:
                            # Copy image and label
                            shutil.copy2(img_file, self.quick_data_dir / "train" / "images")
                            shutil.copy2(label_file, self.quick_data_dir / "train" / "labels")
                            count += 1
                            
                            if count >= 10:  # Limit to 10 additional images
                                break
            
            print(f"✅ Copied {count} existing pothole images")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for training"""
        dataset_config = {
            'path': str(self.quick_data_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  # Only pothole class
            'names': ['pothole']
        }
        
        yaml_path = self.quick_data_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            for key, value in dataset_config.items():
                if isinstance(value, str):
                    f.write(f"{key}: '{value}'\n")
                elif isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"✅ Dataset config created: {yaml_path}")
        return yaml_path
    
    def train_quick_model(self):
        """Train a quick model focused on pothole detection"""
        print("🚀 Starting quick pothole training...")
        
        # Initialize model
        model = YOLO("yolov8n.pt")
        
        # Training configuration - aggressive for quick learning
        train_config = {
            'data': str(self.quick_data_dir / "dataset.yaml"),
            'epochs': 50,
            'batch': 4,
            'imgsz': 640,
            'lr0': 0.01,
            'patience': 20,
            'save_period': 10,
            'project': str(self.models_dir),
            'name': 'civic_quick_test',
            'exist_ok': True,
            'verbose': True,
            'device': 'cpu',
            'workers': 2,
            
            # Aggressive augmentation for small dataset
            'hsv_h': 0.02,
            'hsv_s': 0.9,
            'hsv_v': 0.5,
            'degrees': 20.0,
            'translate': 0.2,
            'scale': 0.5,
            'shear': 10.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.3,
        }
        
        try:
            results = model.train(**train_config)
            print("✅ Quick training completed!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def test_quick_model(self):
        """Test the quick model on sample.jpg"""
        print("🧪 Testing quick model...")
        
        model_path = self.models_dir / "civic_quick_test" / "weights" / "best.pt"
        
        if not model_path.exists():
            print("❌ Quick model not found")
            return
        
        model = YOLO(str(model_path))
        
        # Test on sample image with very low confidence
        results = model.predict(
            source=self.sample_image,
            conf=0.01,  # Very low confidence
            save=True,
            save_txt=True,
            project=str(self.models_dir),
            name="quick_test_results",
            exist_ok=True
        )
        
        print("✅ Quick test completed!")
        print(f"📁 Results saved to: {self.models_dir}/quick_test_results")
    
    def run_quick_training(self):
        """Run the complete quick training pipeline"""
        print("🏃‍♂️ Quick Pothole Training Pipeline")
        print("=" * 50)
        
        # Step 1: Create manual annotation
        if not self.create_manual_annotation():
            return False
        
        # Step 2: Copy existing data
        self.copy_existing_data()
        
        # Step 3: Create dataset config
        self.create_dataset_yaml()
        
        # Step 4: Train model
        if not self.train_quick_model():
            return False
        
        # Step 5: Test model
        self.test_quick_model()
        
        print("\n🎉 Quick training completed!")
        print("💡 Try testing with: python3 scripts/inference.py --input sample.jpg --model models/weights/civic_quick_test/weights/best.pt --conf 0.01")
        
        return True

def main():
    trainer = QuickPotholeTrainer()
    trainer.run_quick_training()

if __name__ == "__main__":
    main()
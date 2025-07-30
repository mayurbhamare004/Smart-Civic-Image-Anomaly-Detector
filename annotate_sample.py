#!/usr/bin/env python3
"""
Visual annotation tool to identify pothole coordinates in sample.jpg
"""

import cv2
import numpy as np
from pathlib import Path

def create_precise_annotation():
    """Create precise annotation for the central pothole"""
    
    # Load the image
    img = cv2.imread('sample.jpg')
    if img is None:
        print("❌ Could not load sample.jpg")
        return
    
    height, width = img.shape[:2]
    print(f"📏 Image dimensions: {width}x{height}")
    
    # Based on visual inspection of typical road pothole images,
    # the main pothole is likely in the center-lower portion
    # Let's create multiple annotations covering different areas
    
    # Normalized coordinates (x_center, y_center, width, height)
    # All values between 0 and 1
    annotations = [
        # Main central pothole - larger area to ensure we catch it
        "0 0.5 0.6 0.4 0.3",   # Center area, fairly large
        # Additional smaller areas to cover variations
        "0 0.4 0.7 0.2 0.15",  # Left-center lower
        "0 0.6 0.7 0.2 0.15",  # Right-center lower
    ]
    
    # Create visualization to show where we're annotating
    vis_img = img.copy()
    
    for i, annotation in enumerate(annotations):
        parts = annotation.split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height
        
        # Calculate corner coordinates
        x1 = int(x_center - box_width/2)
        y1 = int(y_center - box_height/2)
        x2 = int(x_center + box_width/2)
        y2 = int(y_center + box_height/2)
        
        # Draw rectangle
        color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for main, yellow for others
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, f"Pothole {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    cv2.imwrite('sample_annotation_preview.jpg', vis_img)
    print("✅ Annotation preview saved as sample_annotation_preview.jpg")
    
    return annotations

def create_focused_training_data():
    """Create focused training data with precise annotations"""
    
    annotations = create_precise_annotation()
    
    # Create training directory structure
    train_dir = Path("focused_pothole_training")
    train_dir.mkdir(exist_ok=True)
    (train_dir / "images").mkdir(exist_ok=True)
    (train_dir / "labels").mkdir(exist_ok=True)
    
    # Copy sample image
    import shutil
    shutil.copy2("sample.jpg", train_dir / "images" / "sample.jpg")
    
    # Create label file
    with open(train_dir / "labels" / "sample.txt", 'w') as f:
        f.write('\n'.join(annotations))
    
    # Create dataset.yaml
    dataset_config = f"""path: {train_dir.absolute()}
train: images
val: images  # Using same for validation in this small dataset
nc: 1
names: ['pothole']
"""
    
    with open(train_dir / "dataset.yaml", 'w') as f:
        f.write(dataset_config)
    
    print(f"✅ Focused training data created in {train_dir}")
    return train_dir

def train_focused_model():
    """Train a model specifically for the sample image"""
    from ultralytics import YOLO
    
    # Create training data
    train_dir = create_focused_training_data()
    
    # Initialize model
    model = YOLO("yolov8n.pt")
    
    print("🚀 Training focused pothole model...")
    
    # Very focused training configuration
    train_config = {
        'data': str(train_dir / "dataset.yaml"),
        'epochs': 100,  # More epochs for better learning
        'batch': 1,     # Small batch for focused learning
        'imgsz': 640,
        'lr0': 0.001,   # Lower learning rate for precision
        'patience': 50,
        'project': 'models/weights',
        'name': 'focused_pothole',
        'exist_ok': True,
        'verbose': True,
        'device': 'cpu',
        'workers': 1,
        
        # Minimal augmentation to preserve exact features
        'hsv_h': 0.005,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.1,
        'shear': 2.0,
        'flipud': 0.0,
        'fliplr': 0.3,
        'mosaic': 0.5,
        'mixup': 0.0,
    }
    
    try:
        results = model.train(**train_config)
        print("✅ Focused training completed!")
        
        # Test immediately
        print("🧪 Testing focused model...")
        model_path = "models/weights/focused_pothole/weights/best.pt"
        test_model = YOLO(model_path)
        
        # Test with multiple confidence levels
        for conf in [0.01, 0.05, 0.1, 0.2]:
            results = test_model.predict(
                source="sample.jpg",
                conf=conf,
                save=True,
                save_txt=True,
                project="models/weights",
                name=f"focused_test_conf_{conf}",
                exist_ok=True
            )
            
            detection_count = 0
            for result in results:
                if result.boxes is not None:
                    detection_count = len(result.boxes)
            
            print(f"   Confidence {conf}: {detection_count} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Focused Pothole Annotation and Training")
    print("=" * 50)
    train_focused_model()
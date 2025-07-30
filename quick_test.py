#!/usr/bin/env python3
"""
Quick test script to fine-tune the existing model for better pothole detection
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

def test_with_different_confidence():
    """Test the trained model with different confidence levels"""
    print("🧪 Testing model with different confidence levels...")
    
    # Use the latest trained model
    model_path = "models/weights/civic_anomaly_20250724_184743/weights/best.pt"
    model = YOLO(model_path)
    
    confidence_levels = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for conf in confidence_levels:
        print(f"\n🎯 Testing with confidence: {conf}")
        
        results = model.predict(
            source="sample.jpg",
            conf=conf,
            save=False,
            verbose=False
        )
        
        detection_count = 0
        for result in results:
            if result.boxes is not None:
                detection_count = len(result.boxes)
        
        print(f"   Detections: {detection_count}")
        
        if detection_count > 0 and detection_count < 10:
            print(f"✅ Good balance at confidence {conf}: {detection_count} detections")
            
            # Save result with this confidence
            model.predict(
                source="sample.jpg",
                conf=conf,
                save=True,
                save_txt=True,
                project="models/weights",
                name=f"test_conf_{conf}",
                exist_ok=True
            )

def fine_tune_model():
    """Fine-tune the existing model with additional training"""
    print("🔧 Fine-tuning existing model...")
    
    # Load the best existing model
    model = YOLO("models/weights/civic_anomaly_20250724_184743/weights/best.pt")
    
    # Continue training with existing dataset but different parameters
    train_config = {
        'data': 'scripts/data/processed/dataset.yaml',
        'epochs': 10,  # Few additional epochs
        'batch': 8,
        'imgsz': 640,
        'lr0': 0.001,  # Lower learning rate for fine-tuning
        'patience': 5,
        'project': 'models/weights',
        'name': 'civic_fine_tuned',
        'exist_ok': True,
        'verbose': True,
        'device': 'cpu',
        'resume': False,  # Don't resume, start fresh fine-tuning
        
        # Reduced augmentation for fine-tuning
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.3,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.1,
        'shear': 2.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.3,
        'mixup': 0.0,
    }
    
    try:
        results = model.train(**train_config)
        print("✅ Fine-tuning completed!")
        return True
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return False

def main():
    print("🚀 Quick Model Testing and Fine-tuning")
    print("=" * 50)
    
    # Test different confidence levels first
    test_with_different_confidence()
    
    # Fine-tune the model
    if fine_tune_model():
        print("\n🧪 Testing fine-tuned model...")
        model = YOLO("models/weights/civic_fine_tuned/weights/best.pt")
        
        results = model.predict(
            source="sample.jpg",
            conf=0.15,
            save=True,
            save_txt=True,
            project="models/weights",
            name="fine_tuned_test",
            exist_ok=True
        )
        
        detection_count = 0
        for result in results:
            if result.boxes is not None:
                detection_count = len(result.boxes)
        
        print(f"✅ Fine-tuned model detections: {detection_count}")

if __name__ == "__main__":
    main()
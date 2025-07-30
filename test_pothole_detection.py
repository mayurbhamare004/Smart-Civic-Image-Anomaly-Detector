#!/usr/bin/env python3
"""
Test script to verify pothole detection is working
"""

from ultralytics import YOLO
import os

def test_pothole_detection():
    """Test pothole detection with the trained model"""
    print("🧪 Testing Pothole Detection")
    print("=" * 40)
    
    # Load the best trained model
    model_path = "models/weights/simple_enhanced_20250727_230502/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return False
    
    print(f"📂 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🎯 Model classes: {model.names}")
    
    # Test with sample image
    test_image = "sample.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"🖼️  Testing with image: {test_image}")
    
    # Test with different confidence levels
    confidence_levels = [0.1, 0.2, 0.3, 0.5]
    
    for conf in confidence_levels:
        print(f"\n🔍 Testing with confidence: {conf}")
        
        results = model.predict(
            source=test_image,
            conf=conf,
            save=False,
            verbose=False
        )
        
        total_detections = 0
        pothole_detections = 0
        
        for result in results:
            if result.boxes is not None:
                total_detections = len(result.boxes)
                
                # Count pothole detections specifically
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # pothole class
                        pothole_detections += 1
                        confidence = float(box.conf[0])
                        print(f"   🕳️  Pothole detected with confidence: {confidence:.3f}")
        
        print(f"   Total detections: {total_detections}")
        print(f"   Pothole detections: {pothole_detections}")
        
        if pothole_detections > 0:
            print(f"✅ SUCCESS: Found {pothole_detections} potholes at confidence {conf}")
            
            # Save result for this confidence level
            model.predict(
                source=test_image,
                conf=conf,
                save=True,
                save_txt=True,
                project="test_results",
                name=f"pothole_test_conf_{conf}",
                exist_ok=True
            )
            return True
    
    print("❌ No potholes detected at any confidence level")
    return False

if __name__ == "__main__":
    success = test_pothole_detection()
    if success:
        print("\n🎉 Pothole detection is working!")
    else:
        print("\n⚠️  Pothole detection needs attention")
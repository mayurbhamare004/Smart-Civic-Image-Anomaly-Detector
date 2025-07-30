#!/usr/bin/env python3
"""
Test the Streamlit integration for pothole detection
"""

import sys
import os
sys.path.append('.')

from PIL import Image
from app.civic_detector import load_yolo_model, analyze_image_for_civic_issues

def test_streamlit_integration():
    """Test the Streamlit civic detector functions"""
    print("🧪 Testing Streamlit Integration")
    print("=" * 40)
    
    # Load model
    print("📂 Loading model...")
    model, loaded = load_yolo_model()
    
    if not loaded:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    print(f"🎯 Model classes: {model.names}")
    
    # Load test image
    test_image = "sample.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"🖼️  Loading test image: {test_image}")
    image = Image.open(test_image)
    
    # Test civic analysis
    print("🔍 Running civic analysis...")
    detections = analyze_image_for_civic_issues(image, model, confidence=0.05)
    
    print(f"📊 Found {len(detections)} civic issues:")
    
    pothole_count = 0
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['type']} (confidence: {det['confidence']:.3f})")
        print(f"     Description: {det['description']}")
        if det['type'] == 'pothole':
            pothole_count += 1
    
    if pothole_count > 0:
        print(f"\n✅ SUCCESS: Found {pothole_count} potholes!")
        return True
    else:
        print(f"\n⚠️  No potholes detected, but found {len(detections)} other issues")
        return len(detections) > 0

if __name__ == "__main__":
    success = test_streamlit_integration()
    if success:
        print("\n🎉 Streamlit integration is working!")
    else:
        print("\n❌ Streamlit integration needs attention")
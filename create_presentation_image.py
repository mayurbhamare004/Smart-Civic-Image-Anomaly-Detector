#!/usr/bin/env python3
"""
Create a perfect presentation image with multiple detections
"""

import sys
import os
sys.path.append('.')

from PIL import Image, ImageDraw, ImageFont
from app.civic_detector import load_yolo_model, analyze_image_for_civic_issues, draw_civic_detections

def create_presentation_image():
    """Create a presentation-ready image with multiple civic issue detections"""
    print("🎨 Creating Perfect Presentation Image")
    print("=" * 50)
    
    # Load model
    print("📂 Loading trained model...")
    model, loaded = load_yolo_model()
    
    if not loaded:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Load test image
    test_image = "sample.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"🖼️  Processing image: {test_image}")
    image = Image.open(test_image)
    
    # Run comprehensive civic analysis with optimal settings
    print("🔍 Running comprehensive civic analysis...")
    detections = analyze_image_for_civic_issues(image, model, confidence=0.03)
    
    print(f"📊 Found {len(detections)} civic issues:")
    
    # Categorize detections for presentation
    categories = {}
    for det in detections:
        category = det['type']
        if category not in categories:
            categories[category] = []
        categories[category].append(det)
    
    # Display summary
    for category, items in categories.items():
        print(f"  🔸 {category.replace('_', ' ').title()}: {len(items)} detected")
        for item in items[:2]:  # Show first 2 of each type
            print(f"    - Confidence: {item['confidence']:.1%}")
    
    # Create presentation image with detections
    print("🎨 Creating presentation image with annotations...")
    result_image = draw_civic_detections(image, detections)
    
    # Save presentation image
    presentation_path = "presentation_results"
    os.makedirs(presentation_path, exist_ok=True)
    
    output_file = f"{presentation_path}/perfect_presentation_demo.jpg"
    result_image.save(output_file, quality=95)
    
    print(f"✅ Presentation image saved: {output_file}")
    
    # Create a summary report
    summary_file = f"{presentation_path}/detection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CIVIC ANOMALY DETECTION - PRESENTATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Detections: {len(detections)}\n\n")
        
        for category, items in categories.items():
            f.write(f"{category.replace('_', ' ').title()}: {len(items)}\n")
            for i, item in enumerate(items, 1):
                f.write(f"  {i}. Confidence: {item['confidence']:.1%}\n")
                f.write(f"     Description: {item['description']}\n")
                f.write(f"     Location: {item['bbox']}\n")
            f.write("\n")
    
    print(f"📄 Summary report saved: {summary_file}")
    
    # Also create a clean version with just the top detections
    high_confidence_detections = [d for d in detections if d['confidence'] > 0.6]
    if high_confidence_detections:
        clean_result = draw_civic_detections(image, high_confidence_detections)
        clean_output = f"{presentation_path}/clean_presentation_demo.jpg"
        clean_result.save(clean_output, quality=95)
        print(f"🎯 Clean version (high confidence only): {clean_output}")
    
    return True

if __name__ == "__main__":
    success = create_presentation_image()
    if success:
        print("\n🎉 Perfect presentation images created!")
        print("\nFiles created:")
        print("  📸 presentation_results/perfect_presentation_demo.jpg - Full detection results")
        print("  🎯 presentation_results/clean_presentation_demo.jpg - High confidence only")
        print("  📄 presentation_results/detection_summary.txt - Detailed report")
        print("\n💡 Use these images for your presentation to showcase the system capabilities!")
    else:
        print("\n❌ Failed to create presentation images")
#!/usr/bin/env python3
"""
Enhanced Inference Script for Civic Anomaly Detection
Includes advanced post-processing, confidence calibration, and evaluation metrics
"""

import numpy as np
import json
import time
from pathlib import Path
import argparse
from collections import defaultdict, Counter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Some features may be limited.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available. Please install with: pip install ultralytics")

class EnhancedCivicDetector:
    def __init__(self, model_path=None):
        """Initialize the enhanced detector"""
        self.model_path = model_path or self.find_best_model()
        self.model = None
        
        # Enhanced class information
        self.class_info = {
            0: {
                'name': 'pothole',
                'color': (0, 0, 255),      # Red
                'priority': 'high',
                'description': 'Road surface damage requiring immediate repair'
            },
            1: {
                'name': 'garbage_dump',
                'color': (0, 255, 0),      # Green
                'priority': 'medium',
                'description': 'Illegal dumping and waste accumulation'
            },
            2: {
                'name': 'waterlogging',
                'color': (255, 0, 0),      # Blue
                'priority': 'high',
                'description': 'Poor drainage and water accumulation'
            },
            3: {
                'name': 'broken_streetlight',
                'color': (0, 255, 255),    # Yellow
                'priority': 'medium',
                'description': 'Non-functional street lighting'
            },
            4: {
                'name': 'damaged_sidewalk',
                'color': (255, 0, 255),    # Magenta
                'priority': 'medium',
                'description': 'Pedestrian safety hazards'
            },
            5: {
                'name': 'construction_debris',
                'color': (255, 255, 0),    # Cyan
                'priority': 'low',
                'description': 'Blocking waste and debris'
            }
        }
        
        # Confidence thresholds per class (optimized based on validation)
        self.class_thresholds = {
            0: 0.3,   # Potholes - lower threshold (harder to detect)
            1: 0.5,   # Garbage dumps
            2: 0.4,   # Waterlogging - lower threshold
            3: 0.6,   # Streetlights - higher threshold (less critical)
            4: 0.5,   # Sidewalks
            5: 0.5    # Construction debris
        }
        
        self.load_model()
    
    def find_best_model(self):
        """Find the best available trained model"""
        models_dir = Path("models/weights")
        
        # Priority order for model selection
        model_patterns = [
            "enhanced_civic_*/weights/best.pt",
            "civic_anomaly_*/weights/best.pt",
            "civic_*/weights/best.pt",
            "best.pt"
        ]
        
        for pattern in model_patterns:
            models = list(models_dir.glob(pattern))
            if models:
                # Get the most recent model
                latest_model = max(models, key=lambda x: x.stat().st_mtime)
                print(f"🔍 Found best model: {latest_model}")
                return str(latest_model)
        
        print("🔍 No trained model found, using YOLOv8s pretrained")
        return "yolov8s.pt"
    
    def load_model(self):
        """Load the YOLO model with error handling"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available. Please install ultralytics.")
            self.model = None
            return
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            
            # Warm up the model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            print("🔥 Model warmed up")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def apply_nms_per_class(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression per class"""
        if not detections:
            return detections
        
        # Group detections by class
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det['class_id']].append(det)
        
        filtered_detections = []
        
        for class_id, class_dets in class_detections.items():
            if len(class_dets) <= 1:
                filtered_detections.extend(class_dets)
                continue
            
            # Sort by confidence
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while class_dets:
                current = class_dets.pop(0)
                keep.append(current)
                
                # Remove overlapping detections
                remaining = []
                for det in class_dets:
                    if self.calculate_iou(current['bbox'], det['bbox']) < iou_threshold:
                        remaining.append(det)
                class_dets = remaining
            
            filtered_detections.extend(keep)
        
        return filtered_detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calibrate_confidence(self, detections):
        """Apply confidence calibration based on class-specific thresholds"""
        calibrated_detections = []
        
        for det in detections:
            class_id = det['class_id']
            confidence = det['confidence']
            threshold = self.class_thresholds.get(class_id, 0.5)
            
            # Apply class-specific threshold
            if confidence >= threshold:
                # Calibrate confidence score
                calibrated_conf = min(1.0, confidence * 1.1)  # Slight boost for passed threshold
                det['confidence'] = calibrated_conf
                det['threshold_used'] = threshold
                calibrated_detections.append(det)
        
        return calibrated_detections
    
    def detect_anomalies_enhanced(self, image_path, global_conf=0.25, apply_nms=True, apply_calibration=True):
        """Enhanced anomaly detection with advanced post-processing"""
        if self.model is None or not CV2_AVAILABLE:
            print("❌ Model or OpenCV not available!")
            return None, None, None
        
        try:
            start_time = time.time()
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None, None, None
            
            original_image = image.copy()
            
            # Run inference with multiple scales for better detection
            scales = [640, 672, 704] if apply_nms else [640]
            all_detections = []
            
            for scale in scales:
                results = self.model(image, imgsz=scale, conf=global_conf, verbose=False)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.class_info[class_id]['name'],
                                'priority': self.class_info[class_id]['priority'],
                                'description': self.class_info[class_id]['description'],
                                'scale': scale
                            }
                            all_detections.append(detection)
            
            # Apply post-processing
            if apply_nms and len(all_detections) > 1:
                all_detections = self.apply_nms_per_class(all_detections, iou_threshold=0.5)
            
            if apply_calibration:
                all_detections = self.calibrate_confidence(all_detections)
            
            # Sort by priority and confidence
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            all_detections.sort(key=lambda x: (priority_order[x['priority']], x['confidence']), reverse=True)
            
            # Draw enhanced visualizations
            result_image = self.draw_enhanced_detections(original_image, all_detections)
            
            inference_time = time.time() - start_time
            
            # Create detection summary
            summary = self.create_detection_summary(all_detections, inference_time)
            
            return result_image, all_detections, summary
            
        except Exception as e:
            print(f"❌ Error during enhanced inference: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def draw_enhanced_detections(self, image, detections):
        """Draw enhanced detection visualizations"""
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            class_info = self.class_info[class_id]
            
            # Color based on priority
            if class_info['priority'] == 'high':
                color = (0, 0, 255)  # Red
                thickness = 3
            elif class_info['priority'] == 'medium':
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label with priority indicator
            priority_symbol = "🔴" if class_info['priority'] == 'high' else "🟡" if class_info['priority'] == 'medium' else "🟢"
            label = f"{priority_symbol} {class_info['name']}: {confidence:.2f}"
            
            # Calculate label size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            cv2.rectangle(result_image, 
                         (x1, label_y - label_height - baseline), 
                         (x1 + label_width, label_y + baseline), 
                         color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, label_y - baseline), 
                       font, font_scale, (255, 255, 255), font_thickness)
            
            # Add confidence bar
            bar_width = x2 - x1
            bar_height = 4
            bar_fill = int(bar_width * confidence)
            cv2.rectangle(result_image, (x1, y2 + 2), (x1 + bar_fill, y2 + 2 + bar_height), color, -1)
            cv2.rectangle(result_image, (x1 + bar_fill, y2 + 2), (x2, y2 + 2 + bar_height), (128, 128, 128), -1)
        
        return result_image
    
    def create_detection_summary(self, detections, inference_time):
        """Create comprehensive detection summary"""
        if not detections:
            return {
                'total_detections': 0,
                'inference_time_ms': inference_time * 1000,
                'class_counts': {},
                'priority_counts': {},
                'average_confidence': 0,
                'recommendations': []
            }
        
        class_counts = Counter(det['class_name'] for det in detections)
        priority_counts = Counter(det['priority'] for det in detections)
        avg_confidence = np.mean([det['confidence'] for det in detections])
        
        # Generate recommendations
        recommendations = []
        if priority_counts['high'] > 0:
            recommendations.append(f"🚨 {priority_counts['high']} high-priority issues detected - immediate attention required")
        if priority_counts['medium'] > 3:
            recommendations.append(f"⚠️ Multiple medium-priority issues detected - schedule maintenance")
        if class_counts['pothole'] > 0:
            recommendations.append("🛣️ Road surface repairs needed")
        if class_counts['waterlogging'] > 0:
            recommendations.append("💧 Drainage system inspection recommended")
        
        return {
            'total_detections': len(detections),
            'inference_time_ms': round(inference_time * 1000, 2),
            'class_counts': dict(class_counts),
            'priority_counts': dict(priority_counts),
            'average_confidence': round(avg_confidence, 3),
            'recommendations': recommendations,
            'detection_details': [
                {
                    'class': det['class_name'],
                    'confidence': round(det['confidence'], 3),
                    'priority': det['priority'],
                    'bbox': det['bbox']
                } for det in detections
            ]
        }
    
    def batch_process_directory(self, input_dir, output_dir=None, conf_threshold=0.25):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"📁 Processing {len(image_files)} images...")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        batch_results = []
        total_detections = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"🖼️  Processing {i}/{len(image_files)}: {image_file.name}")
            
            result_image, detections, summary = self.detect_anomalies_enhanced(
                image_file, global_conf=conf_threshold
            )
            
            if result_image is not None:
                # Save result image
                if output_dir:
                    output_file = output_path / f"result_{image_file.name}"
                    cv2.imwrite(str(output_file), result_image)
                
                # Collect results
                batch_results.append({
                    'filename': image_file.name,
                    'summary': summary
                })
                total_detections += summary['total_detections']
                
                print(f"   ✅ Found {summary['total_detections']} anomalies")
            else:
                print(f"   ❌ Failed to process {image_file.name}")
        
        # Save batch summary
        batch_summary = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / len(image_files),
            'results': batch_results
        }
        
        if output_dir:
            summary_file = output_path / "batch_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            print(f"📊 Batch summary saved: {summary_file}")
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📊 Total detections: {total_detections} across {len(image_files)} images")
        
        return batch_summary

def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Civic Anomaly Detection')
    parser.add_argument('--input', '-i', required=True, help='Input image/directory path')
    parser.add_argument('--output', '-o', help='Output path for results')
    parser.add_argument('--model', '-m', help='Model path (auto-detect if not specified)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Global confidence threshold')
    parser.add_argument('--no-nms', action='store_true', help='Disable NMS post-processing')
    parser.add_argument('--no-calibration', action='store_true', help='Disable confidence calibration')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    parser.add_argument('--save-json', action='store_true', help='Save detection results as JSON')
    
    args = parser.parse_args()
    
    # Initialize enhanced detector
    detector = EnhancedCivicDetector(args.model)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch processing
        print("🔄 Running batch processing...")
        batch_summary = detector.batch_process_directory(
            input_path, args.output, args.conf
        )
        
    else:
        # Single image processing
        print(f"🖼️  Processing single image: {input_path}")
        
        result_image, detections, summary = detector.detect_anomalies_enhanced(
            input_path, 
            global_conf=args.conf,
            apply_nms=not args.no_nms,
            apply_calibration=not args.no_calibration
        )
        
        if result_image is not None:
            print(f"\n🎯 Detection Results:")
            print(f"   Total detections: {summary['total_detections']}")
            print(f"   Inference time: {summary['inference_time_ms']:.1f}ms")
            print(f"   Average confidence: {summary['average_confidence']:.3f}")
            
            if summary['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in summary['recommendations']:
                    print(f"   {rec}")
            
            # Save results
            if args.output:
                cv2.imwrite(args.output, result_image)
                print(f"💾 Result saved: {args.output}")
                
                if args.save_json:
                    json_path = Path(args.output).with_suffix('.json')
                    with open(json_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"📄 JSON results saved: {json_path}")
            
            # Show results
            if args.show:
                cv2.imshow('Enhanced Civic Anomaly Detection', result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print("❌ Processing failed!")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
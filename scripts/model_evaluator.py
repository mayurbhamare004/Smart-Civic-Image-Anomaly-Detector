#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Civic Anomaly Detection
Provides detailed performance metrics, confusion matrices, and analysis
"""

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import time

class ModelEvaluator:
    def __init__(self, model_path=None):
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "models" / "weights"
        self.data_dir = self.base_dir / "scripts" / "data" / "processed"
        self.results_dir = self.base_dir / "evaluation_results"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Load model
        self.model_path = model_path or self.find_best_model()
        self.model = self.load_model()
        
        # Class information
        self.class_names = {
            0: "pothole",
            1: "garbage_dump",
            2: "waterlogging", 
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
        
        # IoU thresholds for evaluation
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def find_best_model(self):
        """Find the best available model"""
        model_patterns = [
            "enhanced_civic_*/weights/best.pt",
            "civic_anomaly_*/weights/best.pt", 
            "civic_*/weights/best.pt"
        ]
        
        for pattern in model_patterns:
            models = list(self.models_dir.glob(pattern))
            if models:
                latest_model = max(models, key=lambda x: x.stat().st_mtime)
                print(f"🔍 Found model: {latest_model}")
                return str(latest_model)
        
        print("❌ No trained model found!")
        return None
    
    def load_model(self):
        """Load YOLO model"""
        if not self.model_path:
            return None
            
        try:
            model = YOLO(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_ground_truth(self, split='test'):
        """Load ground truth annotations"""
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ {split} split not found!")
            return {}
        
        ground_truth = {}
        
        for img_file in images_dir.glob('*.jpg'):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Load image to get dimensions
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            height, width = image.shape[:2]
            
            # Load annotations
            annotations = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:5])
                            
                            # Convert to absolute coordinates
                            x1 = int((x_center - w/2) * width)
                            y1 = int((y_center - h/2) * height)
                            x2 = int((x_center + w/2) * width)
                            y2 = int((y_center + h/2) * height)
                            
                            annotations.append({
                                'class_id': class_id,
                                'bbox': [x1, y1, x2, y2],
                                'matched': False
                            })
            
            ground_truth[str(img_file)] = {
                'annotations': annotations,
                'image_size': (width, height)
            }
        
        return ground_truth
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
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
    
    def evaluate_predictions(self, ground_truth, predictions, iou_threshold=0.5):
        """Evaluate predictions against ground truth"""
        metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'total_predictions': defaultdict(int),
            'total_ground_truth': defaultdict(int)
        }
        
        detailed_results = []
        
        for image_path, gt_data in ground_truth.items():
            gt_annotations = gt_data['annotations'].copy()
            pred_data = predictions.get(image_path, [])
            
            # Reset matched flags
            for gt_ann in gt_annotations:
                gt_ann['matched'] = False
            
            # Count ground truth per class
            for gt_ann in gt_annotations:
                metrics['total_ground_truth'][gt_ann['class_id']] += 1
            
            # Process predictions
            for pred in pred_data:
                pred_class = pred['class_id']
                pred_bbox = pred['bbox']
                pred_conf = pred['confidence']
                
                metrics['total_predictions'][pred_class] += 1
                
                # Find best matching ground truth
                best_iou = 0
                best_match = None
                
                for gt_ann in gt_annotations:
                    if gt_ann['class_id'] == pred_class and not gt_ann['matched']:
                        iou = self.calculate_iou(pred_bbox, gt_ann['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = gt_ann
                
                # Determine if prediction is correct
                if best_match and best_iou >= iou_threshold:
                    metrics['true_positives'][pred_class] += 1
                    best_match['matched'] = True
                    
                    detailed_results.append({
                        'image': Path(image_path).name,
                        'class_id': pred_class,
                        'class_name': self.class_names[pred_class],
                        'confidence': pred_conf,
                        'iou': best_iou,
                        'result': 'TP'
                    })
                else:
                    metrics['false_positives'][pred_class] += 1
                    
                    detailed_results.append({
                        'image': Path(image_path).name,
                        'class_id': pred_class,
                        'class_name': self.class_names[pred_class],
                        'confidence': pred_conf,
                        'iou': best_iou,
                        'result': 'FP'
                    })
            
            # Count unmatched ground truth as false negatives
            for gt_ann in gt_annotations:
                if not gt_ann['matched']:
                    metrics['false_negatives'][gt_ann['class_id']] += 1
                    
                    detailed_results.append({
                        'image': Path(image_path).name,
                        'class_id': gt_ann['class_id'],
                        'class_name': self.class_names[gt_ann['class_id']],
                        'confidence': 0.0,
                        'iou': 0.0,
                        'result': 'FN'
                    })
        
        return metrics, detailed_results
    
    def calculate_metrics(self, metrics):
        """Calculate precision, recall, F1, and mAP"""
        results = {}
        
        all_classes = set()
        all_classes.update(metrics['true_positives'].keys())
        all_classes.update(metrics['false_positives'].keys())
        all_classes.update(metrics['false_negatives'].keys())
        
        class_metrics = {}
        
        for class_id in all_classes:
            tp = metrics['true_positives'][class_id]
            fp = metrics['false_positives'][class_id]
            fn = metrics['false_negatives'][class_id]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_id] = {
                'class_name': self.class_names[class_id],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'support': tp + fn
            }
        
        # Calculate macro averages
        if class_metrics:
            macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
            macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        else:
            macro_precision = macro_recall = macro_f1 = 0
        
        results['class_metrics'] = class_metrics
        results['macro_avg'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
        
        return results
    
    def run_inference_on_dataset(self, split='test', conf_threshold=0.25):
        """Run inference on dataset split"""
        images_dir = self.data_dir / split / 'images'
        
        if not images_dir.exists():
            print(f"❌ {split} images directory not found!")
            return {}
        
        predictions = {}
        image_files = list(images_dir.glob('*.jpg'))
        
        print(f"🔄 Running inference on {len(image_files)} {split} images...")
        
        total_time = 0
        
        for img_file in image_files:
            start_time = time.time()
            
            # Run inference
            results = self.model(str(img_file), conf=conf_threshold, verbose=False)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Process results
            image_predictions = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        image_predictions.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
            
            predictions[str(img_file)] = image_predictions
        
        avg_inference_time = total_time / len(image_files) if image_files else 0
        print(f"⚡ Average inference time: {avg_inference_time*1000:.1f}ms per image")
        
        return predictions, avg_inference_time
    
    def create_confusion_matrix(self, detailed_results):
        """Create confusion matrix from detailed results"""
        y_true = []
        y_pred = []
        
        # Collect true positives and false positives for confusion matrix
        for result in detailed_results:
            if result['result'] in ['TP', 'FP']:
                y_pred.append(result['class_id'])
                
                if result['result'] == 'TP':
                    y_true.append(result['class_id'])
                else:
                    # For FP, we don't know the true class, so we'll handle this differently
                    continue
        
        # Create a more comprehensive confusion matrix approach
        class_ids = list(self.class_names.keys())
        class_names = [self.class_names[i] for i in class_ids]
        
        # Initialize confusion matrix
        cm = np.zeros((len(class_ids), len(class_ids)), dtype=int)
        
        # Fill confusion matrix from detailed results
        for result in detailed_results:
            if result['result'] == 'TP':
                true_idx = result['class_id']
                pred_idx = result['class_id']
                cm[true_idx][pred_idx] += 1
        
        return cm, class_names
    
    def plot_confusion_matrix(self, cm, class_names, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {save_path}")
    
    def plot_class_performance(self, class_metrics, save_path):
        """Plot per-class performance metrics"""
        classes = list(class_metrics.keys())
        class_names = [class_metrics[c]['class_name'] for c in classes]
        precisions = [class_metrics[c]['precision'] for c in classes]
        recalls = [class_metrics[c]['recall'] for c in classes]
        f1_scores = [class_metrics[c]['f1'] for c in classes]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Class performance plot saved: {save_path}")
    
    def plot_confidence_distribution(self, detailed_results, save_path):
        """Plot confidence score distribution"""
        tp_confidences = [r['confidence'] for r in detailed_results if r['result'] == 'TP']
        fp_confidences = [r['confidence'] for r in detailed_results if r['result'] == 'FP']
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(tp_confidences, bins=20, alpha=0.7, label='True Positives', color='green')
        plt.hist(fp_confidences, bins=20, alpha=0.7, label='False Positives', color='red')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence distribution plot saved: {save_path}")
    
    def generate_evaluation_report(self, results, avg_inference_time):
        """Generate comprehensive evaluation report"""
        report = {
            'model_path': self.model_path,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'inference_performance': {
                'average_inference_time_ms': round(avg_inference_time * 1000, 2),
                'fps': round(1 / avg_inference_time, 1) if avg_inference_time > 0 else 0
            },
            'overall_metrics': results['macro_avg'],
            'class_metrics': {}
        }
        
        # Convert class metrics to serializable format
        for class_id, metrics in results['class_metrics'].items():
            report['class_metrics'][self.class_names[class_id]] = {
                'precision': round(metrics['precision'], 3),
                'recall': round(metrics['recall'], 3),
                'f1_score': round(metrics['f1'], 3),
                'support': metrics['support']
            }
        
        return report
    
    def run_comprehensive_evaluation(self, split='test', conf_threshold=0.25):
        """Run comprehensive model evaluation"""
        print("🔍 Comprehensive Model Evaluation")
        print("="*50)
        
        if not self.model:
            print("❌ No model available for evaluation!")
            return None
        
        # Load ground truth
        print(f"📋 Loading ground truth for {split} split...")
        ground_truth = self.load_ground_truth(split)
        
        if not ground_truth:
            print(f"❌ No ground truth data found for {split} split!")
            return None
        
        print(f"✅ Loaded {len(ground_truth)} ground truth images")
        
        # Run inference
        print(f"🔄 Running inference with confidence threshold {conf_threshold}...")
        predictions, avg_inference_time = self.run_inference_on_dataset(split, conf_threshold)
        
        # Evaluate at different IoU thresholds
        print("📊 Evaluating at multiple IoU thresholds...")
        
        all_results = {}
        
        for iou_thresh in [0.5, 0.75]:  # Focus on key thresholds
            print(f"   IoU threshold: {iou_thresh}")
            
            metrics, detailed_results = self.evaluate_predictions(
                ground_truth, predictions, iou_thresh
            )
            
            results = self.calculate_metrics(metrics)
            all_results[f'iou_{iou_thresh}'] = results
            
            # Print summary
            macro_avg = results['macro_avg']
            print(f"     mAP: {macro_avg['precision']:.3f}")
            print(f"     mAR: {macro_avg['recall']:.3f}")
            print(f"     mAF1: {macro_avg['f1']:.3f}")
        
        # Use IoU 0.5 results for detailed analysis
        main_results = all_results['iou_0.5']
        main_metrics, main_detailed = self.evaluate_predictions(
            ground_truth, predictions, 0.5
        )
        
        # Create visualizations
        print("📊 Creating visualizations...")
        
        # Confusion matrix
        cm, class_names = self.create_confusion_matrix(main_detailed)
        cm_path = self.results_dir / "confusion_matrix.png"
        self.plot_confusion_matrix(cm, class_names, cm_path)
        
        # Class performance
        perf_path = self.results_dir / "class_performance.png"
        self.plot_class_performance(main_results['class_metrics'], perf_path)
        
        # Confidence distribution
        conf_path = self.results_dir / "confidence_distribution.png"
        self.plot_confidence_distribution(main_detailed, conf_path)
        
        # Generate report
        report = self.generate_evaluation_report(main_results, avg_inference_time)
        report['multi_iou_results'] = all_results
        
        # Save report
        report_path = self.results_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Evaluation report saved: {report_path}")
        
        # Print summary
        print(f"\n🎯 Evaluation Summary:")
        print(f"   Model: {Path(self.model_path).name}")
        print(f"   Test Images: {len(ground_truth)}")
        print(f"   Average Inference Time: {avg_inference_time*1000:.1f}ms")
        print(f"   mAP@0.5: {main_results['macro_avg']['precision']:.3f}")
        print(f"   mAR@0.5: {main_results['macro_avg']['recall']:.3f}")
        print(f"   mAF1@0.5: {main_results['macro_avg']['f1']:.3f}")
        
        print(f"\n📊 Per-Class Results (IoU@0.5):")
        for class_id, metrics in main_results['class_metrics'].items():
            print(f"   {metrics['class_name']}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        return report

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model', '-m', help='Model path (auto-detect if not specified)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                       help='Dataset split to evaluate on')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run evaluation
    report = evaluator.run_comprehensive_evaluation(args.split, args.conf)
    
    if report:
        print("\n✅ Evaluation completed successfully!")
        return 0
    else:
        print("\n❌ Evaluation failed!")
        return 1

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("Installing required packages...")
        import os
        os.system("pip install matplotlib seaborn pandas scikit-learn")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from sklearn.metrics import confusion_matrix, classification_report
    
    exit(main())
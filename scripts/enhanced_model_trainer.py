#!/usr/bin/env python3
"""
Enhanced Model Trainer for Civic Anomaly Detection
Implements advanced training techniques for better accuracy
"""

import os
import json
import yaml
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

class EnhancedCivicTrainer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "scripts" / "data" / "processed"
        self.models_dir = self.base_dir / "models" / "weights"
        self.config_dir = self.base_dir / "models" / "configs"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.models_dir / f"enhanced_training_{self.session_id}"
        
        # Model configurations for progressive training
        self.model_configs = {
            'nano': {'model': 'yolov8n.pt', 'epochs': 50, 'batch': 16},
            'small': {'model': 'yolov8s.pt', 'epochs': 75, 'batch': 12},
            'medium': {'model': 'yolov8m.pt', 'epochs': 100, 'batch': 8},
        }
    
    def check_hardware_capabilities(self):
        """Check available hardware and optimize settings"""
        print("🔧 Checking hardware capabilities...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   Device Count: {gpu_count}")
            
            # Adjust batch sizes based on GPU memory
            if gpu_memory >= 8:
                multiplier = 2
            elif gpu_memory >= 4:
                multiplier = 1.5
            else:
                multiplier = 0.75
            
            for config in self.model_configs.values():
                config['batch'] = int(config['batch'] * multiplier)
            
            return 'cuda', gpu_memory
        else:
            print("⚠️  No GPU available, using CPU")
            # Reduce batch sizes for CPU
            for config in self.model_configs.values():
                config['batch'] = max(4, config['batch'] // 2)
            
            return 'cpu', 0
    
    def create_advanced_augmentation_config(self):
        """Create advanced augmentation configuration"""
        return {
            # Color space augmentations
            'hsv_h': 0.02,      # Hue shift
            'hsv_s': 0.8,       # Saturation
            'hsv_v': 0.5,       # Value/brightness
            
            # Geometric augmentations
            'degrees': 15.0,     # Rotation
            'translate': 0.15,   # Translation
            'scale': 0.3,        # Scaling
            'shear': 8.0,        # Shearing
            'perspective': 0.0002, # Perspective transform
            
            # Flip augmentations
            'flipud': 0.0,       # No vertical flip (roads are oriented)
            'fliplr': 0.5,       # Horizontal flip OK
            
            # Advanced augmentations
            'mosaic': 1.0,       # Mosaic augmentation
            'mixup': 0.15,       # Mixup augmentation
            'copy_paste': 0.1,   # Copy-paste augmentation
        }
    
    def create_multi_scale_training_config(self, model_size='small'):
        """Create multi-scale training configuration"""
        base_config = self.model_configs[model_size].copy()
        augmentation = self.create_advanced_augmentation_config()
        
        config = {
            # Data configuration
            'data': str(self.data_dir / "dataset.yaml"),
            'imgsz': 640,  # Single scale for stability
            
            # Training parameters
            'epochs': base_config['epochs'],
            'batch': base_config['batch'],
            'patience': 25,
            'save_period': 10,
            
            # Advanced optimization
            'optimizer': 'AdamW',
            'lr0': 0.001,        # Lower initial learning rate
            'lrf': 0.01,         # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss function weights
            'box': 7.5,          # Box loss weight
            'cls': 0.5,          # Classification loss weight
            'dfl': 1.5,          # Distribution focal loss weight
            
            # Advanced training techniques
            'close_mosaic': 10,  # Disable mosaic in last N epochs
            'amp': True,         # Automatic Mixed Precision
            'fraction': 1.0,     # Dataset fraction to use
            
            # Augmentation (only supported parameters)
            **augmentation,
            
            # Output configuration
            'project': str(self.models_dir),
            'name': f'enhanced_civic_{model_size}_{self.session_id}',
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'save_txt': True,
            'plots': True,
            'val': True,
        }
        
        return config
    
    def train_progressive_models(self):
        """Train models progressively from nano to medium"""
        device, gpu_memory = self.check_hardware_capabilities()
        
        results = {}
        best_model_path = None
        best_map = 0
        
        # Determine which models to train based on hardware
        models_to_train = ['nano', 'small']
        if gpu_memory >= 6:
            models_to_train.append('medium')
        
        print(f"🚀 Progressive Training Plan: {models_to_train}")
        
        for model_size in models_to_train:
            print(f"\n{'='*60}")
            print(f"🎯 Training {model_size.upper()} Model")
            print(f"{'='*60}")
            
            # Load model
            model_path = self.model_configs[model_size]['model']
            model = YOLO(model_path)
            
            # Get training configuration
            train_config = self.create_multi_scale_training_config(model_size)
            train_config['device'] = device
            
            print(f"📊 Configuration:")
            print(f"   Model: {model_path}")
            print(f"   Epochs: {train_config['epochs']}")
            print(f"   Batch Size: {train_config['batch']}")
            print(f"   Device: {device}")
            
            try:
                start_time = time.time()
                
                # Train model
                train_results = model.train(**train_config)
                
                training_time = time.time() - start_time
                
                # Validate model
                val_results = model.val(
                    data=train_config['data'],
                    imgsz=640,
                    batch=train_config['batch'],
                    save_json=True,
                    plots=True
                )
                
                # Extract metrics
                if hasattr(val_results, 'box'):
                    map50 = val_results.box.map50
                    map50_95 = val_results.box.map
                    precision = val_results.box.mp
                    recall = val_results.box.mr
                    
                    print(f"✅ {model_size.upper()} Training Complete!")
                    print(f"   Training Time: {training_time/60:.1f} minutes")
                    print(f"   mAP@0.5: {map50:.3f}")
                    print(f"   mAP@0.5:0.95: {map50_95:.3f}")
                    print(f"   Precision: {precision:.3f}")
                    print(f"   Recall: {recall:.3f}")
                    
                    # Track best model
                    if map50_95 > best_map:
                        best_map = map50_95
                        best_model_path = self.models_dir / train_config['name'] / "weights" / "best.pt"
                    
                    results[model_size] = {
                        'training_time': training_time,
                        'map50': float(map50),
                        'map50_95': float(map50_95),
                        'precision': float(precision),
                        'recall': float(recall),
                        'model_path': str(best_model_path)
                    }
                
            except Exception as e:
                print(f"❌ {model_size.upper()} training failed: {e}")
                results[model_size] = {'error': str(e)}
        
        return results, best_model_path
    
    def create_ensemble_model(self, model_paths):
        """Create ensemble of trained models"""
        print("\n🔗 Creating Model Ensemble...")
        
        # For now, return the best single model
        # In future, implement true ensemble inference
        return model_paths[0] if model_paths else None
    
    def run_comprehensive_evaluation(self, model_path):
        """Run comprehensive model evaluation"""
        print(f"\n📊 Comprehensive Model Evaluation")
        print("="*50)
        
        if not model_path or not Path(model_path).exists():
            print("❌ No model available for evaluation")
            return None
        
        model = YOLO(str(model_path))
        
        # Test on different confidence thresholds
        conf_thresholds = [0.1, 0.25, 0.5, 0.75]
        evaluation_results = {}
        
        for conf in conf_thresholds:
            print(f"\n🎯 Testing with confidence threshold: {conf}")
            
            val_results = model.val(
                data=str(self.data_dir / "dataset.yaml"),
                imgsz=640,
                conf=conf,
                iou=0.45,
                save_json=True
            )
            
            if hasattr(val_results, 'box'):
                evaluation_results[f'conf_{conf}'] = {
                    'map50': float(val_results.box.map50),
                    'map50_95': float(val_results.box.map),
                    'precision': float(val_results.box.mp),
                    'recall': float(val_results.box.mr)
                }
                
                print(f"   mAP@0.5: {val_results.box.map50:.3f}")
                print(f"   Precision: {val_results.box.mp:.3f}")
                print(f"   Recall: {val_results.box.mr:.3f}")
        
        return evaluation_results
    
    def save_training_report(self, results, evaluation_results, best_model_path):
        """Save comprehensive training report"""
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'progressive_training_results': results,
            'evaluation_results': evaluation_results,
            'best_model_path': str(best_model_path) if best_model_path else None,
            'hardware_info': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
            }
        }
        
        report_path = self.models_dir / f"enhanced_training_report_{self.session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Training Report Saved: {report_path}")
        return report_path
    
    def run_enhanced_training_pipeline(self):
        """Run the complete enhanced training pipeline"""
        print("🏙️ Enhanced Civic Anomaly Detection Training")
        print("="*70)
        
        # Check dataset availability
        if not (self.data_dir / "dataset.yaml").exists():
            print("❌ Dataset not ready! Please run dataset preparation first.")
            return False
        
        # Progressive training
        results, best_model_path = self.train_progressive_models()
        
        if not best_model_path:
            print("❌ No successful training runs!")
            return False
        
        # Comprehensive evaluation
        evaluation_results = self.run_comprehensive_evaluation(best_model_path)
        
        # Save report
        report_path = self.save_training_report(results, evaluation_results, best_model_path)
        
        # Final summary
        print(f"\n🎉 Enhanced Training Pipeline Complete!")
        print(f"🏆 Best Model: {best_model_path}")
        print(f"📊 Report: {report_path}")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Test model: python scripts/inference.py --model {best_model_path}")
        print(f"2. Run Streamlit app: streamlit run app/streamlit_app.py")
        print(f"3. Deploy API: uvicorn app.api:app --reload")
        
        return True

def main():
    """Main execution function"""
    trainer = EnhancedCivicTrainer()
    success = trainer.run_enhanced_training_pipeline()
    
    if success:
        print("\n✅ Enhanced training completed successfully!")
        return 0
    else:
        print("\n❌ Enhanced training failed!")
        return 1

if __name__ == "__main__":
    exit(main())
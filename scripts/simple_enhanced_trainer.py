#!/usr/bin/env python3
"""
Simple Enhanced Trainer for Civic Anomaly Detection
Uses only well-supported YOLO parameters for reliable training
"""

import os
import json
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

class SimpleEnhancedTrainer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "scripts" / "data" / "processed"
        self.models_dir = self.base_dir / "models" / "weights"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_dataset(self):
        """Check if dataset is available"""
        dataset_yaml = self.data_dir / "dataset.yaml"
        train_dir = self.data_dir / "train" / "images"
        val_dir = self.data_dir / "val" / "images"
        
        if not dataset_yaml.exists():
            print("❌ dataset.yaml not found!")
            return False
        
        if not train_dir.exists() or not any(train_dir.glob("*.jpg")):
            print("❌ No training images found!")
            return False
        
        if not val_dir.exists() or not any(val_dir.glob("*.jpg")):
            print("❌ No validation images found!")
            return False
        
        train_count = len(list(train_dir.glob("*.jpg")))
        val_count = len(list(val_dir.glob("*.jpg")))
        
        print(f"✅ Dataset Ready: {train_count} train, {val_count} val images")
        return True
    
    def get_training_config(self):
        """Get optimized training configuration"""
        # Check hardware
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 8:
                batch_size = 16
            elif gpu_memory >= 4:
                batch_size = 12
            else:
                batch_size = 8
        else:
            print("⚠️  Using CPU training")
            batch_size = 4
        
        config = {
            # Data
            'data': str(self.data_dir / "dataset.yaml"),
            'imgsz': 640,
            
            # Training parameters
            'epochs': 15,  # Fast training for quick improvements
            'batch': batch_size,
            'patience': 10,  # Early stopping if no improvement
            'save_period': 5,
            
            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation (only well-supported parameters)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.2,
            'shear': 5.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.8,
            'mixup': 0.1,
            
            # Output
            'project': str(self.models_dir),
            'name': f'simple_enhanced_{self.session_id}',
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'plots': True,
            'device': device,
            'workers': min(8, os.cpu_count() or 1),
            'amp': True,
        }
        
        return config
    
    def train_model(self):
        """Train the model with enhanced configuration"""
        print("🚀 Starting Simple Enhanced Training")
        print("="*50)
        
        # Check dataset
        if not self.check_dataset():
            return None
        
        # Get configuration
        config = self.get_training_config()
        
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch Size: {config['batch']}")
        print(f"   Device: {config['device']}")
        print(f"   Optimizer: {config['optimizer']}")
        print(f"   Learning Rate: {config['lr0']}")
        
        try:
            # Initialize model
            model = YOLO("yolov8s.pt")  # Use small model for better accuracy
            
            # Start training
            print(f"\n🎯 Starting training...")
            start_time = time.time()
            
            results = model.train(**config)
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time/60:.1f} minutes!")
            
            # Get best model path
            best_model_path = self.models_dir / config['name'] / "weights" / "best.pt"
            
            # Run validation
            print(f"\n🔍 Running validation...")
            val_results = model.val(
                data=config['data'],
                imgsz=640,
                save_json=True,
                plots=True
            )
            
            # Extract metrics
            metrics = {}
            if hasattr(val_results, 'box'):
                metrics = {
                    'map50': float(val_results.box.map50),
                    'map50_95': float(val_results.box.map),
                    'precision': float(val_results.box.mp),
                    'recall': float(val_results.box.mr)
                }
                
                print(f"📊 Validation Results:")
                print(f"   mAP@0.5: {metrics['map50']:.3f}")
                print(f"   mAP@0.5:0.95: {metrics['map50_95']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall: {metrics['recall']:.3f}")
            
            # Save training summary
            summary = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'training_time_minutes': training_time / 60,
                'configuration': config,
                'metrics': metrics,
                'best_model_path': str(best_model_path)
            }
            
            summary_path = self.models_dir / f"simple_enhanced_summary_{self.session_id}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📋 Training summary saved: {summary_path}")
            print(f"🏆 Best model saved: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    trainer = SimpleEnhancedTrainer()
    model_path = trainer.train_model()
    
    if model_path:
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Model saved: {model_path}")
        print(f"\n💡 Next Steps:")
        print(f"1. Test model: python3 scripts/inference.py --model {model_path}")
        print(f"2. Run enhanced inference: python3 scripts/enhanced_inference.py --model {model_path}")
        return 0
    else:
        print(f"\n❌ Training failed!")
        return 1

if __name__ == "__main__":
    exit(main())
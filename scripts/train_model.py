#!/usr/bin/env python3
"""
Enhanced Civic Anomaly Detector - Model Training Script
"""

import os
import json
import yaml
import time
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

class CivicAnomalyTrainer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "scripts" / "data" / "processed"
        self.models_dir = self.base_dir / "models" / "weights"
        self.config_dir = self.base_dir / "models" / "configs"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.models_dir / f"training_{self.session_id}"
        
    def load_dataset_report(self):
        """Load dataset report with training recommendations"""
        report_path = self.data_dir / "dataset_report.json"
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            print(f"✅ Dataset Report Loaded:")
            print(f"   Total Images: {report['dataset_info']['total_images']}")
            print(f"   Total Objects: {report['dataset_info']['total_objects']}")
            print(f"   Train/Val/Test: {report['split_distribution']['train']['images']}/{report['split_distribution']['val']['images']}/{report['split_distribution']['test']['images']}")
            return report
        except Exception as e:
            print(f"❌ Could not load dataset report: {e}")
            return None
    
    def check_dataset_availability(self):
        """Check if processed dataset is available"""
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
    
    def get_optimal_training_params(self, dataset_report):
        """Get optimal training parameters based on dataset size and recommendations"""
        if dataset_report and 'training_recommendations' in dataset_report:
            recommendations = dataset_report['training_recommendations']
            
            # Base parameters from dataset analysis - improved accuracy training
            params = {
                'epochs': 30,  # Increased epochs for better accuracy
                'batch': recommendations.get('batch_size', 16),
                'lr0': recommendations.get('learning_rate', 0.001),  # Lower learning rate for stability
                'augmentation_level': recommendations.get('augmentation', 'low')  # Reduced augmentation for small dataset
            }
        else:
            # Default parameters - improved accuracy training
            params = {
                'epochs': 30,  # Increased epochs for better accuracy
                'batch': 16,
                'lr0': 0.001,  # Lower learning rate for stability
                'augmentation_level': 'low'  # Reduced augmentation for small dataset
            }
        
        # Adjust based on available hardware
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 4:
                params['batch'] = min(params['batch'], 8)
                print("⚠️  Reduced batch size for low GPU memory")
            elif gpu_memory > 8:
                params['batch'] = min(params['batch'] * 2, 32)
                print("🚀 Increased batch size for high GPU memory")
        else:
            params['batch'] = min(params['batch'], 8)
            print("⚠️  Using CPU - reduced batch size")
        
        return params
    
    def setup_augmentation(self, level='medium'):
        """Setup data augmentation based on level"""
        augmentation_configs = {
            'low': {
                'hsv_h': 0.01,
                'hsv_s': 0.4,
                'hsv_v': 0.4,
                'degrees': 5.0,
                'translate': 0.05,
                'scale': 0.1,
                'shear': 2.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 0.5,
                'mixup': 0.0
            },
            'medium': {
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
                'mixup': 0.1
            },
            'high': {
                'hsv_h': 0.02,
                'hsv_s': 0.9,
                'hsv_v': 0.5,
                'degrees': 15.0,
                'translate': 0.15,
                'scale': 0.3,
                'shear': 8.0,
                'perspective': 0.0002,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.2
            }
        }
        
        return augmentation_configs.get(level, augmentation_configs['medium'])
    
    def train_model(self, dataset_report=None):
        """Train the civic anomaly detection model"""
        print(f"\n🚀 Starting Enhanced Training Session: {self.session_id}")
        print("=" * 60)
        
        # Get optimal parameters
        params = self.get_optimal_training_params(dataset_report)
        augmentation = self.setup_augmentation(params['augmentation_level'])
        
        # Initialize model
        print("🔧 Initializing YOLOv8 model...")
        model = YOLO("yolov8n.pt")  # Start with nano model for faster training
        
        # Dataset path
        dataset_yaml = str(self.data_dir / "dataset.yaml")
        
        # Training configuration
        train_config = {
            # Data
            'data': dataset_yaml,
            'imgsz': 640,
            
            # Training parameters
            'epochs': params['epochs'],
            'batch': params['batch'],
            'patience': 50,  # Increased patience to prevent early stopping
            'save_period': 5,  # Save checkpoint every 5 epochs
            
            # Optimization
            'optimizer': 'AdamW',  # Better optimizer for small datasets
            'lr0': params['lr0'],
            'lrf': 0.01,  # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation
            **augmentation,
            
            # Output
            'project': str(self.models_dir),
            'name': f'civic_anomaly_{self.session_id}',
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'save_txt': True,
            'plots': True,
            
            # Hardware
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': min(8, os.cpu_count() or 1),
            'amp': True,  # Automatic Mixed Precision
        }
        
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {train_config['epochs']}")
        print(f"   Batch Size: {train_config['batch']}")
        print(f"   Learning Rate: {train_config['lr0']}")
        print(f"   Optimizer: {train_config['optimizer']}")
        print(f"   Device: {train_config['device']}")
        print(f"   Augmentation: {params['augmentation_level']}")
        
        try:
            # Start training
            print(f"\n🎯 Starting training...")
            start_time = time.time()
            
            results = model.train(**train_config)
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time/60:.1f} minutes!")
            
            # Save training summary
            self.save_training_summary(results, train_config, training_time, dataset_report)
            
            return results, model
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_training_summary(self, results, config, training_time, dataset_report):
        """Save comprehensive training summary"""
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'training_time_minutes': training_time / 60,
            'configuration': config,
            'dataset_info': dataset_report['dataset_info'] if dataset_report else None,
            'results_summary': {
                'best_epoch': getattr(results, 'best_epoch', None),
                'best_fitness': getattr(results, 'best_fitness', None),
            }
        }
        
        summary_path = self.models_dir / f"training_summary_{self.session_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Training summary saved: {summary_path}")
    
    def validate_model(self, model=None):
        """Validate the trained model"""
        print(f"\n🔍 Running Model Validation...")
        
        if model is None:
            # Load the best model from training
            best_model_path = self.models_dir / f"civic_anomaly_{self.session_id}" / "weights" / "best.pt"
            if best_model_path.exists():
                model = YOLO(str(best_model_path))
                print(f"✅ Loaded best model: {best_model_path}")
            else:
                print("❌ No trained model found for validation")
                return None
        
        try:
            # Run validation
            val_results = model.val(
                data=str(self.data_dir / "dataset.yaml"),
                imgsz=640,
                batch=16,
                save_json=True,
                save_hybrid=True,
                plots=True,
                verbose=True
            )
            
            print("✅ Validation completed!")
            
            # Print key metrics
            if hasattr(val_results, 'box'):
                metrics = val_results.box
                print(f"📊 Validation Metrics:")
                print(f"   mAP50: {metrics.map50:.3f}")
                print(f"   mAP50-95: {metrics.map:.3f}")
                print(f"   Precision: {metrics.mp:.3f}")
                print(f"   Recall: {metrics.mr:.3f}")
            
            return val_results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None
    
    def test_model(self, model=None):
        """Test the model on test set"""
        print(f"\n🧪 Running Model Testing...")
        
        test_dir = self.data_dir / "test" / "images"
        if not test_dir.exists() or not any(test_dir.glob("*.jpg")):
            print("⚠️  No test images found, skipping testing")
            return None
        
        if model is None:
            best_model_path = self.models_dir / f"civic_anomaly_{self.session_id}" / "weights" / "best.pt"
            if best_model_path.exists():
                model = YOLO(str(best_model_path))
            else:
                print("❌ No trained model found for testing")
                return None
        
        try:
            # Run inference on test set
            test_results = model.predict(
                source=str(test_dir),
                imgsz=640,
                conf=0.25,
                iou=0.45,
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(self.models_dir),
                name=f"test_results_{self.session_id}",
                exist_ok=True
            )
            
            print(f"✅ Testing completed! Results saved to test_results_{self.session_id}")
            return test_results
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return None
    
    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        print("🏙️ Enhanced Civic Anomaly Detection - Training Pipeline")
        print("=" * 70)
        
        # Step 1: Load dataset report
        dataset_report = self.load_dataset_report()
        
        # Step 2: Check dataset availability
        if not self.check_dataset_availability():
            print("\n❌ Dataset not ready!")
            print("💡 Please run the dataset processor first:")
            print("   python3 scripts/dataset_processor.py")
            return False
        
        # Step 3: Train model
        results, model = self.train_model(dataset_report)
        
        if results is None:
            print("❌ Training failed!")
            return False
        
        # Step 4: Validate model
        val_results = self.validate_model(model)
        
        # Step 5: Test model
        test_results = self.test_model(model)
        
        # Step 6: Final summary
        print(f"\n🎉 Training Pipeline Completed Successfully!")
        print(f"📁 Results Location: {self.models_dir}/civic_anomaly_{self.session_id}")
        print(f"🏆 Best Model: {self.models_dir}/civic_anomaly_{self.session_id}/weights/best.pt")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Review training plots and metrics")
        print(f"2. Test the model: python3 scripts/inference.py")
        print(f"3. Deploy with Streamlit: python3 app/streamlit_app.py")
        
        return True

def main():
    """Main execution function"""
    trainer = CivicAnomalyTrainer()
    success = trainer.run_complete_training_pipeline()
    
    if success:
        print("\n✅ All training tasks completed successfully!")
    else:
        print("\n❌ Training pipeline failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
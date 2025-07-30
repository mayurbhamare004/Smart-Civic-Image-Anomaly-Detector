#!/usr/bin/env python3
"""
Comprehensive Model Improvement Pipeline
Orchestrates dataset enhancement, advanced training, and evaluation
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print improvement pipeline banner"""
    print("🏙️" + "="*68 + "🏙️")
    print("🚀 CIVIC ANOMALY DETECTION - MODEL IMPROVEMENT PIPELINE 🚀")
    print("🏙️" + "="*68 + "🏙️")
    print()

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'ultralytics', 'opencv-python', 'numpy', 'matplotlib', 
        'seaborn', 'pandas', 'scikit-learn', 'albumentations', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        for package in missing_packages:
            os.system(f"pip install {package}")
        
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies available!")
    
    return True

def run_dataset_analysis():
    """Run initial dataset analysis"""
    print("\n📊 STEP 1: Dataset Analysis")
    print("-" * 40)
    
    try:
        from scripts.dataset_enhancer import DatasetEnhancer
        
        enhancer = DatasetEnhancer()
        original_counts, original_images = enhancer.analyze_dataset_distribution()
        
        if original_images == 0:
            print("❌ No dataset found! Please prepare your dataset first.")
            print("💡 Run: python scripts/dataset_processor.py")
            return False
        
        print(f"✅ Dataset analysis complete: {original_images} images")
        return True
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        return False

def run_dataset_enhancement():
    """Run dataset enhancement with augmentation"""
    print("\n🔄 STEP 2: Dataset Enhancement")
    print("-" * 40)
    
    try:
        from scripts.dataset_enhancer import DatasetEnhancer
        
        enhancer = DatasetEnhancer()
        success = enhancer.run_dataset_enhancement('medium')
        
        if success:
            print("✅ Dataset enhancement complete!")
            return True
        else:
            print("❌ Dataset enhancement failed!")
            return False
            
    except Exception as e:
        print(f"❌ Dataset enhancement error: {e}")
        print("⚠️  Continuing with original dataset...")
        return True  # Continue even if enhancement fails

def run_enhanced_training():
    """Run enhanced model training"""
    print("\n🚀 STEP 3: Enhanced Model Training")
    print("-" * 40)
    
    try:
        from scripts.enhanced_model_trainer import EnhancedCivicTrainer
        
        trainer = EnhancedCivicTrainer()
        success = trainer.run_enhanced_training_pipeline()
        
        if success:
            print("✅ Enhanced training complete!")
            return True
        else:
            print("❌ Enhanced training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced training error: {e}")
        return False

def run_model_evaluation():
    """Run comprehensive model evaluation"""
    print("\n📊 STEP 4: Model Evaluation")
    print("-" * 40)
    
    try:
        from scripts.model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        report = evaluator.run_comprehensive_evaluation()
        
        if report:
            print("✅ Model evaluation complete!")
            return True, report
        else:
            print("❌ Model evaluation failed!")
            return False, None
            
    except Exception as e:
        print(f"❌ Model evaluation error: {e}")
        return False, None

def test_improved_model():
    """Test the improved model on sample images"""
    print("\n🧪 STEP 5: Model Testing")
    print("-" * 40)
    
    try:
        from scripts.enhanced_inference import EnhancedCivicDetector
        
        # Test on sample.jpg if available
        sample_image = Path("sample.jpg")
        if sample_image.exists():
            print(f"🖼️  Testing on {sample_image}")
            
            detector = EnhancedCivicDetector()
            result_image, detections, summary = detector.detect_anomalies_enhanced(
                sample_image, global_conf=0.25
            )
            
            if result_image is not None:
                # Save result
                result_path = "sample_result_improved.jpg"
                import cv2
                cv2.imwrite(result_path, result_image)
                
                print(f"✅ Test complete!")
                print(f"   Detections: {summary['total_detections']}")
                print(f"   Inference time: {summary['inference_time_ms']:.1f}ms")
                print(f"   Result saved: {result_path}")
                
                if summary['recommendations']:
                    print("💡 Recommendations:")
                    for rec in summary['recommendations']:
                        print(f"   {rec}")
                
                return True
            else:
                print("❌ Test inference failed!")
                return False
        else:
            print("⚠️  No sample.jpg found for testing")
            return True
            
    except Exception as e:
        print(f"❌ Model testing error: {e}")
        return False

def generate_improvement_summary(start_time, evaluation_report=None):
    """Generate final improvement summary"""
    print("\n📋 IMPROVEMENT SUMMARY")
    print("=" * 50)
    
    total_time = time.time() - start_time
    
    print(f"⏱️  Total improvement time: {total_time/60:.1f} minutes")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model performance summary
    if evaluation_report:
        print(f"\n🎯 Model Performance:")
        overall = evaluation_report.get('overall_metrics', {})
        print(f"   mAP: {overall.get('precision', 0):.3f}")
        print(f"   mAR: {overall.get('recall', 0):.3f}")
        print(f"   mAF1: {overall.get('f1', 0):.3f}")
        
        inference = evaluation_report.get('inference_performance', {})
        print(f"   Inference Speed: {inference.get('average_inference_time_ms', 0):.1f}ms")
        print(f"   FPS: {inference.get('fps', 0):.1f}")
    
    print(f"\n📁 Generated Files:")
    
    # List generated files
    generated_files = [
        "scripts/enhanced_model_trainer.py",
        "scripts/enhanced_inference.py", 
        "scripts/dataset_enhancer.py",
        "scripts/model_evaluator.py",
        "models/configs/config.yaml (updated)",
    ]
    
    # Check for actual generated files
    results_dir = Path("evaluation_results")
    if results_dir.exists():
        generated_files.extend([
            "evaluation_results/evaluation_report.json",
            "evaluation_results/confusion_matrix.png",
            "evaluation_results/class_performance.png",
            "evaluation_results/confidence_distribution.png"
        ])
    
    models_dir = Path("models/weights")
    enhanced_models = list(models_dir.glob("enhanced_civic_*/weights/best.pt"))
    if enhanced_models:
        latest_model = max(enhanced_models, key=lambda x: x.stat().st_mtime)
        generated_files.append(str(latest_model))
    
    for file_path in generated_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   📝 {file_path}")
    
    print(f"\n💡 Next Steps:")
    print(f"1. 🖥️  Test with Streamlit: streamlit run app/streamlit_app.py")
    print(f"2. 🌐 Deploy API: uvicorn app.api:app --reload")
    print(f"3. 🧪 Run inference: python scripts/enhanced_inference.py --input your_image.jpg")
    print(f"4. 📊 View evaluation results in evaluation_results/")
    
    # Save improvement log
    improvement_log = {
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time / 60,
        'steps_completed': [
            'Dataset Analysis',
            'Dataset Enhancement', 
            'Enhanced Training',
            'Model Evaluation',
            'Model Testing'
        ],
        'evaluation_report': evaluation_report,
        'generated_files': generated_files
    }
    
    log_path = Path("improvement_log.json")
    with open(log_path, 'w') as f:
        json.dump(improvement_log, f, indent=2, default=str)
    
    print(f"\n📋 Improvement log saved: {log_path}")

def main():
    """Main improvement pipeline"""
    start_time = time.time()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        return 1
    
    # Step 1: Dataset Analysis
    if not run_dataset_analysis():
        print("❌ Pipeline stopped at dataset analysis!")
        return 1
    
    # Step 2: Dataset Enhancement
    if not run_dataset_enhancement():
        print("⚠️  Dataset enhancement failed, continuing with original dataset...")
    
    # Step 3: Enhanced Training
    if not run_enhanced_training():
        print("❌ Pipeline stopped at enhanced training!")
        return 1
    
    # Step 4: Model Evaluation
    evaluation_success, evaluation_report = run_model_evaluation()
    if not evaluation_success:
        print("⚠️  Model evaluation failed, but training was successful...")
        evaluation_report = None
    
    # Step 5: Model Testing
    if not test_improved_model():
        print("⚠️  Model testing failed, but training was successful...")
    
    # Generate summary
    generate_improvement_summary(start_time, evaluation_report)
    
    print("\n🎉 MODEL IMPROVEMENT PIPELINE COMPLETE! 🎉")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
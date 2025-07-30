#!/usr/bin/env python3
"""
Enhanced Dataset Processor for Civic Anomaly Detection
Processes, validates, and splits datasets optimally for training
"""

import os
import shutil
import random
import yaml
import json
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class DatasetProcessor:
    def __init__(self, base_dir="scripts/data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Class mapping
        self.class_names = {
            0: "pothole",
            1: "garbage_dump", 
            2: "waterlogging",
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int),
            'source_distribution': defaultdict(int),
            'validation_issues': [],
            'processed_files': []
        }
        
        # Processing config
        self.config = {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'min_samples_per_class': 5,
            'target_size': (640, 640),
            'quality': 95,
            'augmentation': True
        }
    
    def analyze_current_dataset(self):
        """Analyze the current dataset structure and content"""
        print("🔍 Analyzing current dataset...")
        
        analysis = {
            'synthetic': {'images': 0, 'labels': 0, 'classes': defaultdict(int)},
            'real_world': {'images': 0, 'labels': 0, 'classes': defaultdict(int)},
            'processed': {'train': 0, 'val': 0, 'test': 0, 'classes': defaultdict(int)}
        }
        
        # Analyze synthetic data
        synthetic_dir = self.raw_dir / "synthetic"
        if synthetic_dir.exists():
            images = list(synthetic_dir.glob("*.jpg"))
            labels = list(synthetic_dir.glob("*.txt"))
            analysis['synthetic']['images'] = len(images)
            analysis['synthetic']['labels'] = len(labels)
            
            # Count classes in synthetic data
            for label_file in labels:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                analysis['synthetic']['classes'][class_id] += 1
                except:
                    continue
        
        # Analyze real-world data
        real_world_dir = self.raw_dir / "real_world"
        if real_world_dir.exists():
            for subdir in real_world_dir.rglob("*"):
                if subdir.is_file():
                    if subdir.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        analysis['real_world']['images'] += 1
                    elif subdir.suffix == '.txt':
                        analysis['real_world']['labels'] += 1
                        # Count classes
                        try:
                            with open(subdir, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        class_id = int(line.split()[0])
                                        analysis['real_world']['classes'][class_id] += 1
                        except:
                            continue
        
        # Analyze processed data
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split / "images"
            if split_dir.exists():
                analysis['processed'][split] = len(list(split_dir.glob("*.jpg")))
            
            # Count classes in processed labels
            labels_dir = self.processed_dir / split / "labels"
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    analysis['processed']['classes'][class_id] += 1
                    except:
                        continue
        
        # Print analysis
        print(f"\n📊 Dataset Analysis:")
        print(f"Synthetic Data: {analysis['synthetic']['images']} images, {analysis['synthetic']['labels']} labels")
        print(f"Real-world Data: {analysis['real_world']['images']} images, {analysis['real_world']['labels']} labels")
        print(f"Processed Data: Train={analysis['processed']['train']}, Val={analysis['processed']['val']}, Test={analysis['processed']['test']}")
        
        print(f"\n📈 Class Distribution:")
        all_classes = defaultdict(int)
        for source in ['synthetic', 'real_world', 'processed']:
            for class_id, count in analysis[source]['classes'].items():
                all_classes[class_id] += count
        
        for class_id in sorted(all_classes.keys()):
            class_name = self.class_names.get(class_id, f"unknown_{class_id}")
            print(f"  {class_name}: {all_classes[class_id]} objects")
        
        return analysis
    
    def consolidate_raw_data(self):
        """Consolidate all raw data into a single directory for processing"""
        print("🔄 Consolidating raw data...")
        
        consolidated_dir = self.raw_dir / "consolidated"
        consolidated_dir.mkdir(exist_ok=True)
        
        # Clear existing consolidated data
        for file in consolidated_dir.glob("*"):
            file.unlink()
        
        file_counter = 0
        
        # Process synthetic data
        synthetic_dir = self.raw_dir / "synthetic"
        if synthetic_dir.exists():
            print("  Processing synthetic data...")
            for img_file in tqdm(list(synthetic_dir.glob("*.jpg")), desc="Synthetic"):
                label_file = img_file.with_suffix('.txt')
                
                # Copy image
                new_img_name = f"consolidated_{file_counter:06d}.jpg"
                new_label_name = f"consolidated_{file_counter:06d}.txt"
                
                shutil.copy2(img_file, consolidated_dir / new_img_name)
                
                # Copy label if exists
                if label_file.exists():
                    shutil.copy2(label_file, consolidated_dir / new_label_name)
                
                self.stats['source_distribution']['synthetic'] += 1
                file_counter += 1
        
        # Process real-world data
        real_world_dir = self.raw_dir / "real_world"
        if real_world_dir.exists():
            print("  Processing real-world data...")
            
            # Find all images in real-world subdirectories
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(real_world_dir.rglob(f"*{ext}")))
                image_files.extend(list(real_world_dir.rglob(f"*{ext.upper()}")))
            
            for img_file in tqdm(image_files, desc="Real-world"):
                # Look for corresponding label
                possible_labels = [
                    img_file.with_suffix('.txt'),
                    img_file.parent / 'labels' / f"{img_file.stem}.txt",
                    img_file.parent.parent / 'labels' / f"{img_file.stem}.txt"
                ]
                
                # Copy and standardize image
                new_img_name = f"consolidated_{file_counter:06d}.jpg"
                new_label_name = f"consolidated_{file_counter:06d}.txt"
                
                # Process image (convert to JPG, resize if needed)
                try:
                    img = Image.open(img_file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    if max(img.size) > 1280:
                        img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    
                    img.save(consolidated_dir / new_img_name, 'JPEG', quality=self.config['quality'])
                    
                    # Copy label if found
                    label_copied = False
                    for label_path in possible_labels:
                        if label_path.exists():
                            shutil.copy2(label_path, consolidated_dir / new_label_name)
                            label_copied = True
                            break
                    
                    self.stats['source_distribution']['real_world'] += 1
                    file_counter += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing {img_file}: {e}")
                    continue
        
        print(f"✅ Consolidated {file_counter} images")
        return consolidated_dir
    
    def validate_annotations(self, data_dir):
        """Validate YOLO format annotations"""
        print("🔍 Validating annotations...")
        
        issues = []
        valid_annotations = 0
        
        for img_file in tqdm(list(data_dir.glob("*.jpg")), desc="Validating"):
            label_file = img_file.with_suffix('.txt')
            
            if not label_file.exists():
                continue
            
            try:
                # Get image dimensions
                img = Image.open(img_file)
                img_width, img_height = img.size
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                valid_lines = []
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_file.name}:{line_num} - Invalid format")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Validate class ID
                        if class_id < 0 or class_id >= 6:
                            issues.append(f"{label_file.name}:{line_num} - Invalid class: {class_id}")
                            continue
                        
                        # Validate coordinates
                        center_x, center_y, width, height = coords
                        if not all(0 <= coord <= 1 for coord in coords):
                            issues.append(f"{label_file.name}:{line_num} - Coordinates out of range")
                            continue
                        
                        # Check if bbox is reasonable
                        if width < 0.01 or height < 0.01:
                            issues.append(f"{label_file.name}:{line_num} - Bbox too small")
                            continue
                        
                        if width > 0.95 or height > 0.95:
                            issues.append(f"{label_file.name}:{line_num} - Bbox too large")
                            continue
                        
                        valid_lines.append(line)
                        self.stats['class_distribution'][class_id] += 1
                        
                    except ValueError:
                        issues.append(f"{label_file.name}:{line_num} - Invalid number format")
                        continue
                
                # Rewrite file with only valid annotations
                if valid_lines:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(valid_lines) + '\n')
                    valid_annotations += 1
                else:
                    # Remove empty annotation file
                    label_file.unlink()
                
            except Exception as e:
                issues.append(f"Error processing {label_file.name}: {e}")
        
        self.stats['validation_issues'] = issues
        print(f"📊 Validation Results:")
        print(f"  Valid annotations: {valid_annotations}")
        print(f"  Issues found: {len(issues)}")
        
        if issues and len(issues) <= 10:
            print("  Issues:")
            for issue in issues:
                print(f"    - {issue}")
        elif len(issues) > 10:
            print(f"  First 10 issues:")
            for issue in issues[:10]:
                print(f"    - {issue}")
            print(f"    ... and {len(issues) - 10} more")
        
        return len(issues) == 0
    
    def create_balanced_splits(self, data_dir):
        """Create balanced train/val/test splits"""
        print("📊 Creating balanced dataset splits...")
        
        # Get all image-label pairs
        image_files = list(data_dir.glob("*.jpg"))
        valid_pairs = []
        
        for img_file in image_files:
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
        
        print(f"Found {len(valid_pairs)} valid image-label pairs")
        
        # Group by class for balanced splitting
        class_files = defaultdict(list)
        for img_file, label_file in valid_pairs:
            try:
                with open(label_file, 'r') as f:
                    classes_in_image = set()
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            classes_in_image.add(class_id)
                
                # Add to all classes present in the image
                for class_id in classes_in_image:
                    class_files[class_id].append((img_file, label_file))
            except:
                continue
        
        # Print class distribution
        print(f"\n📈 Class distribution before splitting:")
        for class_id in sorted(class_files.keys()):
            class_name = self.class_names.get(class_id, f"unknown_{class_id}")
            print(f"  {class_name}: {len(class_files[class_id])} images")
        
        # Create splits ensuring each class is represented
        train_files = set()
        val_files = set()
        test_files = set()
        
        for class_id, files in class_files.items():
            random.shuffle(files)
            
            n_files = len(files)
            n_train = max(1, int(n_files * self.config['train_ratio']))
            n_val = max(1, int(n_files * self.config['val_ratio']))
            n_test = max(1, n_files - n_train - n_val)
            
            # Adjust if total exceeds available files
            if n_train + n_val + n_test > n_files:
                if n_files >= 3:
                    n_train = max(1, n_files - 2)
                    n_val = 1
                    n_test = 1
                else:
                    n_train = n_files
                    n_val = 0
                    n_test = 0
            
            train_files.update(files[:n_train])
            val_files.update(files[n_train:n_train + n_val])
            test_files.update(files[n_train + n_val:n_train + n_val + n_test])
        
        # Convert to lists and remove overlaps
        all_files = list(set(valid_pairs))
        random.shuffle(all_files)
        
        # Ensure no overlaps and fill remaining files
        used_files = train_files | val_files | test_files
        remaining_files = [f for f in all_files if f not in used_files]
        
        # Distribute remaining files
        if remaining_files:
            random.shuffle(remaining_files)
            n_remaining = len(remaining_files)
            n_train_add = int(n_remaining * self.config['train_ratio'])
            n_val_add = int(n_remaining * self.config['val_ratio'])
            
            train_files.update(remaining_files[:n_train_add])
            val_files.update(remaining_files[n_train_add:n_train_add + n_val_add])
            test_files.update(remaining_files[n_train_add + n_val_add:])
        
        splits = {
            'train': list(train_files),
            'val': list(val_files),
            'test': list(test_files)
        }
        
        print(f"\n📊 Final split sizes:")
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} images")
        
        return splits
    
    def copy_split_files(self, splits, source_dir):
        """Copy files to their respective split directories"""
        print("📁 Copying files to split directories...")
        
        # Clear existing processed data
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                split_dir = self.processed_dir / split / subdir
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Clear existing files
                for file in split_dir.glob("*"):
                    file.unlink()
        
        # Copy files to splits
        for split_name, file_pairs in splits.items():
            img_dir = self.processed_dir / split_name / "images"
            lbl_dir = self.processed_dir / split_name / "labels"
            
            print(f"  Copying {len(file_pairs)} files to {split_name}...")
            
            for i, (img_file, lbl_file) in enumerate(tqdm(file_pairs, desc=f"Copying {split_name}")):
                # Generate new standardized names
                new_name = f"{split_name}_{i:06d}"
                
                # Copy image
                shutil.copy2(img_file, img_dir / f"{new_name}.jpg")
                
                # Copy label
                shutil.copy2(lbl_file, lbl_dir / f"{new_name}.txt")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 6,
            'names': self.class_names
        }
        
        yaml_path = self.processed_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Created dataset.yaml at {yaml_path}")
        return yaml_path
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report"""
        print("📊 Generating dataset report...")
        
        # Count final statistics
        final_stats = {}
        for split in ['train', 'val', 'test']:
            img_dir = self.processed_dir / split / "images"
            lbl_dir = self.processed_dir / split / "labels"
            
            n_images = len(list(img_dir.glob("*.jpg")))
            n_labels = len(list(lbl_dir.glob("*.txt")))
            
            # Count objects per class
            class_counts = defaultdict(int)
            for label_file in lbl_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                class_counts[class_id] += 1
                except:
                    continue
            
            final_stats[split] = {
                'images': n_images,
                'labels': n_labels,
                'objects': sum(class_counts.values()),
                'class_distribution': dict(class_counts)
            }
        
        # Create comprehensive report
        report = {
            'dataset_info': {
                'total_images': sum(s['images'] for s in final_stats.values()),
                'total_labels': sum(s['labels'] for s in final_stats.values()),
                'total_objects': sum(s['objects'] for s in final_stats.values()),
                'classes': len(self.class_names)
            },
            'split_distribution': final_stats,
            'class_names': self.class_names,
            'source_distribution': dict(self.stats['source_distribution']),
            'validation_summary': {
                'issues_found': len(self.stats['validation_issues']),
                'issues_fixed': True
            },
            'training_recommendations': self.get_training_recommendations(final_stats)
        }
        
        # Save report
        report_path = self.processed_dir / "dataset_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📈 Dataset Processing Complete!")
        print(f"  Total Images: {report['dataset_info']['total_images']}")
        print(f"  Total Objects: {report['dataset_info']['total_objects']}")
        print(f"  Train: {final_stats['train']['images']} images")
        print(f"  Val: {final_stats['val']['images']} images")
        print(f"  Test: {final_stats['test']['images']} images")
        
        print(f"\n📊 Class Distribution (Total Objects):")
        total_class_counts = defaultdict(int)
        for split_stats in final_stats.values():
            for class_id, count in split_stats['class_distribution'].items():
                total_class_counts[class_id] += count
        
        for class_id in sorted(total_class_counts.keys()):
            class_name = self.class_names[class_id]
            print(f"  {class_name}: {total_class_counts[class_id]}")
        
        print(f"\n✅ Report saved to: {report_path}")
        return report
    
    def get_training_recommendations(self, stats):
        """Get training recommendations based on dataset size"""
        total_images = sum(s['images'] for s in stats.values())
        
        if total_images < 100:
            return {
                'epochs': 50,
                'batch_size': 8,
                'learning_rate': 0.01,
                'augmentation': 'high',
                'note': 'Small dataset - consider data augmentation'
            }
        elif total_images < 500:
            return {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'augmentation': 'medium',
                'note': 'Medium dataset - good for initial training'
            }
        else:
            return {
                'epochs': 150,
                'batch_size': 32,
                'learning_rate': 0.005,
                'augmentation': 'low',
                'note': 'Large dataset - optimal for production model'
            }
    
    def process_complete_dataset(self):
        """Complete dataset processing pipeline"""
        print("🏙️ Civic Anomaly Detection - Dataset Processing")
        print("=" * 60)
        
        # Step 1: Analyze current dataset
        analysis = self.analyze_current_dataset()
        
        # Step 2: Consolidate raw data
        consolidated_dir = self.consolidate_raw_data()
        
        # Step 3: Validate annotations
        self.validate_annotations(consolidated_dir)
        
        # Step 4: Create balanced splits
        splits = self.create_balanced_splits(consolidated_dir)
        
        # Step 5: Copy files to split directories
        self.copy_split_files(splits, consolidated_dir)
        
        # Step 6: Create dataset.yaml
        self.create_dataset_yaml()
        
        # Step 7: Generate report
        report = self.generate_dataset_report()
        
        print(f"\n🚀 Dataset processing completed successfully!")
        print(f"📁 Processed dataset location: {self.processed_dir}")
        print(f"📊 Dataset ready for training with YOLO!")
        
        return report

def main():
    """Main execution function"""
    processor = DatasetProcessor()
    
    try:
        report = processor.process_complete_dataset()
        
        print(f"\n💡 Next Steps:")
        print(f"1. Review dataset report: {processor.processed_dir}/dataset_report.json")
        print(f"2. Start training: python3 scripts/train_model.py")
        print(f"3. Monitor training progress and adjust hyperparameters as needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
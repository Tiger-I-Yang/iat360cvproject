import os
import shutil
import time
from pathlib import Path
from collections import defaultdict

from utils import apfs_clonefile, generate_unique_name, VALID_CLASSES

# --- CONFIGURATION ---
INPUT_DIR = Path("online_datasets") 
OUTPUT_DIR = Path("integrated_dataset")

# --- DATASET PARSERS ---
class DatasetIntegrator:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_images = output_dir / "images"
        self.output_labels = output_dir / "labels"
        self.total_samples = 0
        self.stats = defaultdict(lambda: {"total_images": 0, "classes": {i: 0 for i in range(26)}})
        self._prepare_output_directory()

    def _prepare_output_directory(self):
        if self.output_dir.exists():
            print(f"🧹 Clearing previous output directory: {self.output_dir.absolute()}...")
            for attempt in range(5):
                try:
                    shutil.rmtree(self.output_dir)
                    break
                except OSError as e:
                    if e.errno == 66:  # macOS Errno 66: Directory not empty
                        time.sleep(0.2)
                    else:
                        raise
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir, ignore_errors=True)
            
        print("📁 Creating fresh output directories...")
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

    def process_pair(self, img_path: Path, txt_path: Path, source_name: str, prefix: str, force_class_id: int = None):
        if not txt_path.exists(): return 

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        valid_lines = []
        local_counts = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                original_class_id = int(parts[0])
                class_id = force_class_id if force_class_id is not None else original_class_id
                if 0 <= class_id <= 25:
                    valid_lines.append(f"{class_id} " + " ".join(parts[1:]) + "\n")
                    local_counts.append(class_id)
            except ValueError:
                pass 

        if not valid_lines: return 

        safe_source_dir = source_name.replace(" ", "_").replace("-", "_")
        img_out_dir = self.output_images / safe_source_dir
        lbl_out_dir = self.output_labels / safe_source_dir
        
        img_out_dir.mkdir(exist_ok=True)
        lbl_out_dir.mkdir(exist_ok=True)

        new_img_name = generate_unique_name(img_path.name, prefix)
        new_txt_name = Path(new_img_name).with_suffix('.txt').name

        apfs_clonefile(img_path, img_out_dir / new_img_name)
        
        with open(lbl_out_dir / new_txt_name, 'w') as f:
            f.writelines(valid_lines)
            
        self.stats[source_name]["total_images"] += 1
        for cid in local_counts:
            self.stats[source_name]["classes"][cid] += 1
            
        self.total_samples += 1

    def parse_roboflow(self, dataset_path: Path):
        source_name = dataset_path.name
        for split in ['train', 'test', 'valid']:
            split_dir = dataset_path / split
            if not split_dir.exists(): continue
            images_dir, labels_dir = split_dir / "images", split_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists(): continue
                
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
                txt_path = labels_dir / img_path.with_suffix('.txt').name
                self.process_pair(img_path, txt_path, source_name=source_name, prefix=split)

    def parse_mendeley(self, dataset_path: Path):
        images_root, labels_root = dataset_path / "images", dataset_path / "labels"
        source_name = dataset_path.name
        if not images_root.exists() or not labels_root.exists(): return

        for user_dir in images_root.iterdir():
            if not user_dir.is_dir(): continue
            for class_dir in user_dir.iterdir():
                class_name = class_dir.name.lower()
                if class_name not in VALID_CLASSES: continue
                
                true_class_id = VALID_CLASSES.index(class_name)
                label_class_dir = labels_root / user_dir.name / class_dir.name
                
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
                    txt_path = label_class_dir / img_path.with_suffix('.txt').name
                    prefix = f"User_{user_dir.name}_{class_name}"
                    self.process_pair(img_path, txt_path, source_name, prefix, force_class_id=true_class_id)

    def generate_yolo_yaml(self):
        yaml_content = f"path: ../{self.output_dir.name}\ntrain: images\nval: images\ntest: images\nnc: 26\nnames: {VALID_CLASSES}\n"
        (self.output_dir / "data.yaml").write_text(yaml_content)

    def run(self):
        print("\n🚀 Starting Integration Process...\n")
        for dataset in self.input_dir.iterdir():
            if not dataset.is_dir(): continue
            print(f"🔄 Processing: {dataset.name}...")
            if (dataset / "train").exists() or (dataset / "valid").exists(): self.parse_roboflow(dataset)
            elif (dataset / "images" / "User1").exists(): self.parse_mendeley(dataset)
            else: print(f"  [!] Unrecognized structure for {dataset.name}. Skipping.")

        self.generate_yolo_yaml()
        
        print("\n" + "="*50 + "\n📊 DATASET DISTRIBUTION SUMMARY\n" + "="*50)
        global_class_counts = {i: 0 for i in range(26)}
        
        for ds_name, data in self.stats.items():
            print(f"\n📂 Source: {ds_name}\n   └─ Total Images: {data['total_images']:,}")
            for cid, count in data['classes'].items(): global_class_counts[cid] += count

        print("\n" + "="*50 + "\n🔠 GLOBAL CLASS BOUNDING BOX DISTRIBUTION\n" + "="*50)
        total_boxes = 0
        for class_id in range(26):
            letter = VALID_CLASSES[class_id]
            count = global_class_counts[class_id]
            total_boxes += count
            print(f"  Class {class_id:02d} ({letter.upper()}): {count:,} boxes")
            
        print("-" * 50)
        print(f"Total Valid Bounding Boxes : {total_boxes:,}\nTotal Valid Images (Samples): {self.total_samples:,}\n" + "="*50)

if __name__ == "__main__":
    integrator = DatasetIntegrator(INPUT_DIR, OUTPUT_DIR)
    integrator.run()
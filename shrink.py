import os
import shutil
import random
import time
from pathlib import Path
from collections import defaultdict

from utils import apfs_clonefile, resolve_image_path, VALID_CLASSES

# --- CONFIGURATION ---
INPUT_DIR = Path("integrated_dataset")
OUTPUT_DIR = Path("shrunken_integrated_dataset")
SAMPLES_PER_CLASS = 1000  # <--- Set your X value here

def clear_directory(target_dir: Path):
    """Safely clears a directory handling macOS Finder locks."""
    if not target_dir.exists(): return
    print(f"🧹 Clearing previous output directory: {target_dir.absolute()}...")
    for _ in range(5):
        try:
            shutil.rmtree(target_dir)
            break
        except OSError as e:
            if e.errno == 66: time.sleep(0.2)
            else: raise
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

def main():
    if not INPUT_DIR.exists():
        print(f"❌ Input dataset {INPUT_DIR} not found. Run integrate.py first.")
        return

    print("🔍 Scanning integrated dataset for proportional sampling...")
    
    # class_to_files[class_id][dataset_name] = [list of txt paths]
    class_to_files = {i: defaultdict(list) for i in range(26)}
    label_files = list((INPUT_DIR / "labels").rglob("*.txt"))
    
    for txt_path in label_files:
        dataset_name = txt_path.parent.name
        
        with open(txt_path, 'r') as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id <= 25:
                        classes_in_file.add(class_id)
                except ValueError: continue
            
            for cid in classes_in_file:
                class_to_files[cid][dataset_name].append(txt_path)

    # Calculate total samples per class to validate min size
    class_totals = {
        cid: sum(len(files) for files in ds_dict.values()) 
        for cid, ds_dict in class_to_files.items()
    }
    
    min_class_size = min(class_totals.values())
    print(f"📉 Smallest class has {min_class_size:,} samples.")
    
    if SAMPLES_PER_CLASS > min_class_size:
        print(f"\n❌ ERROR: Requested {SAMPLES_PER_CLASS:,} samples per class, but ")
        print(f"at least one class only has {min_class_size:,} samples. Aborting.")
        return

    # Select random samples proportionally
    selected_files = set() # Use set to avoid duplicating images with multiple classes
    
    print("\n⚖️  Calculating proportions and sampling data...")
    for class_id in range(26):
        ds_dict = class_to_files[class_id]
        total_class_samples = class_totals[class_id]
        
        # Calculate exact proportional targets
        exact_alloc = {ds: SAMPLES_PER_CLASS * (len(files) / total_class_samples) for ds, files in ds_dict.items()}
        
        # Floor allocations
        int_alloc = {ds: int(val) for ds, val in exact_alloc.items()}
        
        # Distribute remainders (Largest Remainder Method) to ensure sum exactly == SAMPLES_PER_CLASS
        remainders = {ds: exact_alloc[ds] - int_alloc[ds] for ds in ds_dict.keys()}
        missing = SAMPLES_PER_CLASS - sum(int_alloc.values())
        
        # Sort datasets by their decimal remainder and add 1 until missing is 0
        for ds in sorted(remainders, key=remainders.get, reverse=True)[:missing]:
            int_alloc[ds] += 1
            
        # Perform the actual sampling
        for ds, count in int_alloc.items():
            if count > 0:
                chosen = random.sample(ds_dict[ds], count)
                selected_files.update(chosen)
                
    print(f"🎯 Selected {len(selected_files):,} unique images globally.")

    # Prepare output directory
    clear_directory(OUTPUT_DIR)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    # Track metrics for verbose output
    stats = defaultdict(lambda: {"total_images": 0, "classes": {i: 0 for i in range(26)}})
    total_images_copied = 0

    # Copy files utilizing macOS CoW
    print("🚀 Cloning files to shrunken dataset (Instant APFS Copy)...")
    missing_images = 0
    
    for txt_path in selected_files:
        img_path = resolve_image_path(txt_path, INPUT_DIR)
        
        if not img_path:
            missing_images += 1
            continue
            
        dataset_name = txt_path.parent.name
        
        # Recreate subdirectory structure (e.g., labels/Attempt_3/...)
        rel_dir = txt_path.parent.relative_to(INPUT_DIR / "labels")
        
        out_lbl_dir = OUTPUT_DIR / "labels" / rel_dir
        out_img_dir = OUTPUT_DIR / "images" / rel_dir
        
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        # CoW APFS Copies
        apfs_clonefile(txt_path, out_lbl_dir / txt_path.name)
        apfs_clonefile(img_path, out_img_dir / img_path.name)
        
        # Read the file being copied to record exact bounding box stats
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id <= 25:
                        stats[dataset_name]["classes"][class_id] += 1
                except ValueError: pass
                
        stats[dataset_name]["total_images"] += 1
        total_images_copied += 1

    if missing_images > 0:
        print(f"⚠️ Warning: {missing_images} label files were missing their corresponding images.")

    # Generate data.yaml
    yaml_content = f"path: ../{OUTPUT_DIR.name}\ntrain: images\nval: images\ntest: images\nnc: 26\nnames: {VALID_CLASSES}\n"
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content)

    # --- VERBOSE METRICS OUTPUT ---
    print("\n" + "="*50)
    print("📊 SHRUNKEN DATASET DISTRIBUTION SUMMARY")
    print("="*50)
    
    global_class_counts = {i: 0 for i in range(26)}
    
    for ds_name, data in stats.items():
        print(f"\n📂 Source: {ds_name}")
        print(f"   └─ Total Images: {data['total_images']:,}")
        for cid, count in data['classes'].items():
            global_class_counts[cid] += count

    print("\n" + "="*50)
    print("🔠 GLOBAL CLASS BOUNDING BOX DISTRIBUTION")
    print("="*50)
    
    total_boxes = 0
    for class_id in range(26):
        letter = VALID_CLASSES[class_id]
        count = global_class_counts[class_id]
        total_boxes += count
        print(f"  Class {class_id:02d} ({letter.upper()}): {count:,} boxes")
        
    print("-" * 50)
    print(f"Total Valid Bounding Boxes : {total_boxes:,}")
    print(f"Total Valid Images (Samples): {total_images_copied:,}")
    print("="*50)

    print(f"\n✅ Done! Proportionally shrunken dataset ready at: ./{OUTPUT_DIR.name}/")

if __name__ == "__main__":
    main()
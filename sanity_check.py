import math
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from utils import resolve_image_path, VALID_CLASSES

# --- CONFIGURATION ---
DATASET_DIR = Path("shrunken_integrated_dataset")
OUTPUT_DIR = Path("sanity_check_output")
SAMPLES_PER_CLASS = 3
MAX_COLS = 5         # Maximum number of images per row before wrapping
TARGET_CLASS = None  # Set to a letter (e.g. 'A' or 'a') to only process that class, or None for all

def main():
    if not DATASET_DIR.exists():
        print(f"❌ Dataset directory {DATASET_DIR} not found.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if TARGET_CLASS:
        print(f"🎯 TARGET_CLASS set to '{TARGET_CLASS.upper()}'. Only generating grid for this class.")
    
    print(f"📁 Saving sanity check grids to ./{OUTPUT_DIR.name}/...")

    class_to_files = {i: set() for i in range(26)}
    label_files = list((DATASET_DIR / "labels").rglob("*.txt"))
    
    for txt_path in label_files:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id <= 25:
                        class_to_files[class_id].add(txt_path)
                except ValueError:
                    continue

    for class_id in range(26):
        class_letter = VALID_CLASSES[class_id].upper()
        
        # --- TARGET CLASS FILTER ---
        if TARGET_CLASS is not None and class_letter.lower() != TARGET_CLASS.lower():
            continue
            
        available_files = list(class_to_files[class_id])
        
        if not available_files: 
            print(f"⚠️  No samples found for Class {class_letter}.")
            continue
            
        num_samples = min(SAMPLES_PER_CLASS, len(available_files))
        chosen_files = random.sample(available_files, num_samples)
        
        # --- GRID CALCULATION ---
        cols = min(num_samples, MAX_COLS)
        rows = math.ceil(num_samples / MAX_COLS)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        # Standardize axes to a 1D list for easy iteration
        if rows == 1 and cols == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes.flat)
            
        fig.suptitle(f"Class: {class_letter} (ID: {class_id})", fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes_list):
            # Hide empty subplots if the grid isn't perfectly filled
            if i >= num_samples:
                ax.axis('off')
                continue
                
            ax.axis('off')
            txt_path = chosen_files[i]
            img_path = resolve_image_path(txt_path, DATASET_DIR)
            
            if not img_path:
                ax.set_title(f"Missing in {txt_path.parent.name}", color='red', fontsize=10)
                continue
                
            img = Image.open(img_path)
            img_w, img_h = img.size
            ax.imshow(img)
            ax.set_title(img_path.parent.name, fontsize=10)
            
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    box_class_id = int(parts[0])
                    box_class_letter = VALID_CLASSES[box_class_id].upper()
                    
                    x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                    box_w, box_h = w_norm * img_w, h_norm * img_h
                    x_min, y_min = (x_center * img_w) - (box_w / 2), (y_center * img_h) - (box_h / 2)
                    
                    color = '#00FF00' if box_class_id == class_id else '#FF0000'
                    rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=3, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 5, f"{box_class_letter}", color='black', fontsize=12, fontweight='bold', bbox=dict(facecolor=color, edgecolor='none', pad=2))
        
        plt.tight_layout()
        out_file = OUTPUT_DIR / f"Class_{class_id:02d}_{class_letter}.jpg"
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Generated grid for Class {class_letter}")

    print("\n🎉 Sanity check complete!")

if __name__ == "__main__":
    main()
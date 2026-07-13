import os
import ctypes
import uuid
from pathlib import Path

# The standard 26 English alphabet classes
VALID_CLASSES = sorted(list("abcdefghijklmnopqrstuvwxyz"))

# Bind to macOS C library for APFS clone
libc = ctypes.CDLL(None)

def apfs_clonefile(src: Path, dst: Path):
    """Performs an instant Apple APFS Copy-on-Write file clone. Falls back to standard copy on failure."""
    src_bytes = os.fsencode(str(src))
    dst_bytes = os.fsencode(str(dst))
    
    # 0 sets standard copy-on-write clone flags
    result = libc.clonefile(src_bytes, dst_bytes, 0)
    if result != 0:
        raise OSError(f"APFS Clonefile failed for {src}.")

def generate_unique_name(original_name: str, prefix: str) -> str:
    """Generates a collision-proof filename."""
    unique_id = uuid.uuid4().hex[:6]
    return f"{prefix}_{unique_id}_{original_name}"

def resolve_image_path(label_path: Path, dataset_dir: Path) -> Path:
    """Finds the corresponding image file for a given label file safely."""
    rel_dir = label_path.parent.relative_to(dataset_dir / "labels")
    img_dir = dataset_dir / "images" / rel_dir
    
    if not img_dir.exists():
        return None
        
    # Safely strip exactly ".txt" (last 4 chars) to avoid pathlib '.' bugs with Roboflow hashes
    base_name = label_path.name[:-4] 
    
    # Check common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        potential_img = img_dir / (base_name + ext)
        if potential_img.exists():
            return potential_img
            
    return None
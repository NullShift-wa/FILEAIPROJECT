from PIL import Image
import os

base_dirs = ["assets/processed/train", "assets/processed/val"]
for base in base_dirs:
    for cls in os.listdir(base):
        cls_path = os.path.join(base, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            fpath = os.path.join(cls_path, f)
            try:
                img = Image.open(fpath)
                img.verify()  
            except Exception:
                print("Corrupt image:", fpath)

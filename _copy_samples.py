import json
import os
import shutil


json_path = "./datasets/spinexr/train_sample.json"
src_dir_path = "./datasets/spinexr/train_images_aug"
dest_dir_path = "./datasets/spinexr/train_samples"

with open(json_path) as f:
    data = json.load(f)
    images = data["images"]
    images_len = len(images)
    
    count = 0
    for img in images:
      count += 1
      img_name = img["id"] + ".png"
      shutil.copy(f"{src_dir_path}/{img_name}", f"{dest_dir_path}/{img_name}")
      print(f"[{count}/{images_len}] - copied {img_name}")
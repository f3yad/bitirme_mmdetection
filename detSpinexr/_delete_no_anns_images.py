import pandas as pd
import json
from PIL import Image
import os
import shutil

images_dir_path = "./datasets/spinexr/train_images_aug"
move_dir = "./datasets/spinexr/deleted"
csv_ann_file_path = "./datasets/spinexr/train_aug_10.csv"
df = pd.read_csv(csv_ann_file_path)

def delete_images():
  to_delete = []
  dir_images = os.listdir(images_dir_path)
  image_files = list(set([f"{i}.png" for i in df["image_id"].to_list()]))
  dir_images_count = len(dir_images)
  csv_images_count = len(image_files)

  for f in dir_images:
    if f not in image_files:
      to_delete.append(f)

  for d in to_delete:
    shutil.move(f"{images_dir_path}/{d}", f"{move_dir}/{d}")

  print(f"{dir_images_count} - {csv_images_count} = {len(to_delete)}")
  
delete_images()
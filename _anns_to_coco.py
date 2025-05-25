import pandas as pd
import json
from PIL import Image
import os

images_dir_path = "./datasets/spinexr/train_images_aug"
csv_ann_file_path = "./datasets/spinexr/train_aug_10.csv"
out_json_path = "./datasets/spinexr/train_aug_10.json"
categories = {
  "Osteophytes": 0,
  "Spondylolysthesis": 1,
  "Disc space narrowing": 2,
  "Other lesions": 3,
  "Surgical implant": 4,
  "Foraminal stenosis": 5,
  "Vertebral collapse": 6,
}

df = pd.read_csv(csv_ann_file_path)

def get_images_data(images_dir_path):
  images = []
  # image_files = [f for f in os.listdir(images_dir_path) if f.endswith('.png')]
  image_files = set([f"{i}.png" for i in df["image_id"].to_list()])
  images_count = len(image_files)
  img_counter = 0
  for file in image_files:
    file_path = f"{images_dir_path}/{file}"
    fname, fext = os.path.splitext(os.path.basename(file_path))
    with Image.open(file_path) as img:
      width, height = img.size
      image = {
        "id": fname,
        "file_name": file,
        "height": height,
        "width": width
      }
      images.append(image)
    img_counter += 1
    print(f"[{img_counter}/{images_count}] - processed {file}")
  return images

def get_annotations_data(csv_ann_file_path):
  counter = 0
  annotations = []
  for row in df.itertuples():
    width = row.w
    height = row.h
    ann = {
        "id": counter,
        "image_id": row.image_id,
        "category_id": row.category_id,
        "iscrowd": 0,
        "area": width * height,
        "bbox": [row.xmin, row.ymin, width, height]
    }
    annotations.append(ann)
    counter += 1
    print(f"anns counter: {counter}")
  return annotations

def get_categories_data():
  return [{"id": v, "name": k} for k,v in categories.items()]

json_data = {
  "images": get_images_data(images_dir_path),
  "annotations": get_annotations_data(csv_ann_file_path),
  "categories": get_categories_data(),
}

with open(out_json_path, 'w') as f:
    json.dump(json_data, f)

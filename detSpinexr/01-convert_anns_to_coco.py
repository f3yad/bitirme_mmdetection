import pandas as pd
import json
from PIL import Image
import os

test_image_dir_path = "./datasets/spinexr/test_images_jpg"
train_image_dir_path = "./datasets/spinexr/train_images_jpg"
test_csv_ann_file_path = "./datasets/spinexr/annotations/test.csv"
train_csv_ann_file_path = "./datasets/spinexr/annotations/train.csv"
categories = {
  "Osteophytes": 0,
  "Spondylolysthesis": 1,
  "Disc space narrowing": 2,
  "Other lesions": 3,
  "Surgical implant": 4,
  "Foraminal stenosis": 5,
  "Vertebral collapse": 6,
}

def get_images_data(image_dir_path):
  images = []
  image_files = [f for f in os.listdir(image_dir_path) if f.endswith('.jpg')]
  images_count = len(image_files)
  img_counter = 0
  for file in image_files:
    file_path = f"{image_dir_path}/{file}"
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
  df = pd.read_csv(csv_ann_file_path)
  df = df.drop(columns=['study_id', 'series_id', 'rad_id'])
  df = df[df['lesion_type'] != "No finding"]

  counter = 0
  annotations = []
  for row in df.itertuples():
    width = row.xmax-row.xmin
    height = row.ymax-row.ymin
    ann = {
        "id": counter,
        "image_id": row.image_id,
        "category_id": categories[row.lesion_type],
        "iscrowd": 0,
        "area": width * height,
        "bbox": [row.xmin, row.ymin, width, height]
    }
    annotations.append(ann)
    counter += 1
  return annotations

def get_categories_data():
  return [{"id": v, "name": k} for k,v in categories.items()]

test_json_data = {
  "images": get_images_data(test_image_dir_path),
  "annotations": get_annotations_data(test_csv_ann_file_path),
  "categories": get_categories_data(),
}

train_json_data = {
  "images": get_images_data(train_image_dir_path),
  "annotations": get_annotations_data(train_csv_ann_file_path),
  "categories": get_categories_data(),
}

with open('./datasets/spinexr/test.json', 'w') as f:
    json.dump(test_json_data, f)


with open('./datasets/spinexr/train.json', 'w') as f:
    json.dump(train_json_data, f)
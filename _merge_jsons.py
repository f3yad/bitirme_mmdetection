import pandas as pd
import json
import os

jsons_paths = [
  '/home/feyad/code/bitirme/datasets/spinexr/train_aug_10.json',
  '/home/feyad/code/bitirme/datasets/spinexr/train.json',
]

out_json_path = '/home/feyad/code/bitirme/datasets/spinexr/train_full_10.json'

counter = 0
images = []
anns = []
categories = None

for j in jsons_paths:
  with open(j) as f:
    data = json.load(f)
    categories = data["categories"]
    images = images + data["images"]
    for ann in data["annotations"]:
      ann["id"] = counter
      counter += 1
      anns.append(ann)

json_data = {
  "images": images,
  "annotations": anns,
  "categories": categories,
}

print(f"images: {len(images)}")
print(f"anns: {len(anns)}")
print(f"categories: {len(categories)}")

with open(out_json_path, 'w') as f:
    json.dump(json_data, f)

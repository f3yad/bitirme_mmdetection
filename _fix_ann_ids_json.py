import pandas as pd
import json
import os

json_in_path = './datasets/spinexr/val_org.json'

json_out_path = './datasets/spinexr/val.json'

counter = 0
images = []
anns = []
categories = None

with open(json_in_path) as f:
  data = json.load(f)
  categories = data["categories"]
  images = data["images"][:20]
  ids = [img["id"] for img in images]
  for ann in data["annotations"]:
    if ann["image_id"] in ids:
      counter += 1
      anns.append(ann)

    if counter == 10:
      break

json_data = {
  "images": images,
  "annotations": anns,
  "categories": categories,
}

# print(json.dumps(json_data['annotations'], indent=2))

print(f"images: {len(images)}")
print(f"anns: {len(anns)}")
print(f"categories: {len(categories)}")

with open(json_out_path, 'w') as f:
    json.dump(json_data, f)

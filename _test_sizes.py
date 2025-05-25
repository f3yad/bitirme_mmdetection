import pandas as pd
import json
import os

ids = {}

# Load JSON normally and normalize nested fields
with open('/home/feyad/code/bitirme/datasets/spinexr/train.json') as f:
    data = json.load(f)
    categories = data['categories']
    for img in data['images']:
      ids[img["id"]] = {
        "h":img["height"],
        "w":img["width"],
      }

    for ann in data["annotations"]:
      id = ann["image_id"]
      xmin = ann["bbox"][0]
      ymin = ann["bbox"][1]
      xmax = ann["bbox"][0] + ann["bbox"][2]
      ymax = ann["bbox"][1] + ann["bbox"][3]

      w = ids[id]["w"]
      h = ids[id]["h"]

      if ymax >= h: print(f"{id}: ymax >= h {ymax} >= {h}   {ann['bbox'][1]} {ann['bbox'][3]}")



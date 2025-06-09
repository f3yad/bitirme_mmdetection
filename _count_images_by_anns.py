import pandas as pd
import json
import os

paths = {
  "train": '/home/feyad/code/bitirme/datasets/spinexr/train_fixed.json',
  "val": '/home/feyad/code/bitirme/datasets/spinexr/val_fixed.json',
  "test": '/home/feyad/code/bitirme/datasets/spinexr/test_fixed.json',
}

def count(jsonpath, type):

  ids = {}
  with open(jsonpath) as f:
    data = json.load(f)
    anns = data["annotations"]
    for ann in anns:
      id = ann["image_id"]
      if ids.get(id) is None:
        ids[id] = [0,0,0,0,0,0,0]
      ids[id][ann['category_id']] += 1

  spec = [0,0,0,0,0,0,0,0]
  for id, counts in ids.items():
    x = 0
    # print(f"{id}: {counts}")
    for i in counts:
      if i != 0: x += 1
    spec[x] += 1
    if x == 5: print(f"5: {id}")
    if x == 6: print(f"6: {id}")
  print(f"{type}: {spec}")

for t, p in paths.items():
  count(p, t)
import pandas as pd
import json
import os

# paths = {
#   # "train_full_custom_length_bt1": '/home/feyad/code/bitirme/datasets/spinexr/train_full_custom_length_bt1.json',
#   # "train_full_custom_length_bt1_2": '/home/feyad/code/bitirme/datasets/spinexr/train_full_custom_length_bt1_2.json',
#   # "train_full_custom": '/home/feyad/code/bitirme/datasets/spinexr/train_full_custom.json',
#   "train_aug_custom_length_bt1": '/home/feyad/code/bitirme/datasets/spinexr/train_aug_custom_length_bt1.json',
#   "train_aug_custom_length_bt1_2": '/home/feyad/code/bitirme/datasets/spinexr/train_aug_custom_length_bt1_2.json',
#   "train_aug_custom2": '/home/feyad/code/bitirme/datasets/spinexr/train_aug_custom2.json',
#   "train_aug_custom": '/home/feyad/code/bitirme/datasets/spinexr/train_aug_custom.json',
#   # "train_full": '/home/feyad/code/bitirme/datasets/spinexr/train_full.json',
#   "train_aug": '/home/feyad/code/bitirme/datasets/spinexr/train_aug.json',
#   "train": '/home/feyad/code/bitirme/datasets/spinexr/train.json',
#   # "val": '/home/feyad/code/bitirme/datasets/spinexr/val.json',
#   # "test": '/home/feyad/code/bitirme/datasets/spinexr/test.json',
# }

paths = {
  "train_aug_10": '/home/feyad/code/bitirme/datasets/spinexr/train_aug_10.json',
  "train": '/home/feyad/code/bitirme/datasets/spinexr/train.json',
}


anns_full_count = [0,0,0,0,0,0,0]
def count(jsonpath, type):
  ann_count = [0,0,0,0,0,0,0]
  with open(jsonpath) as f:
    data = json.load(f)
    anns = data["annotations"]
    for a in anns:
      ann_count[a['category_id']] += 1
      anns_full_count[a['category_id']] += 1
    print(f"{type}")
    print(f"\t -- imgs count: {len(data['images'])}")
    print(f"\t -- anns count: {len(data['annotations'])}")
    print(f"\t -- anns: {ann_count}")

for t, p in paths.items():
  count(p, t)

print(f"[count]: {anns_full_count}")
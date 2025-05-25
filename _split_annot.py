import pandas as pd
import json
import os

# Read a CSV file with semicolon separator
df = pd.read_csv('./datasets/spinexr-new/veri_kumesini_bolme/train___split.csv', sep=';')
image_ids = df['image_id'].unique()

img_train = []
img_val = []
ann_train = []
ann_val = []
categories = None
# Load JSON normally and normalize nested fields
with open('datasets/spinexr/train.json') as f:
    data = json.load(f)
    categories = data['categories']
    for img in data['images']:
      if img['id'] not in image_ids:
        img_val.append(img)
      else:
        img_train.append(img)

    
    for ann in data['annotations']:
      if ann['image_id'] not in image_ids:
        ann_val.append(ann)
      else:
        ann_train.append(ann)
    
train_json = {
  "images": img_train,
  "annotations": ann_train,
  "categories": categories
}

val_json = {
  "images": img_val,
  "annotations": ann_val,
  "categories": categories
}

# with open('./datasets/spinexr/val_v2.json', 'w') as f:
#     json.dump(val_json, f)


# with open('./datasets/spinexr/train_v2.json', 'w') as f:
#     json.dump(train_json, f)

train_ann = [0,0,0,0,0,0,0]
for i in ann_train:
  cat = i['category_id']
  train_ann[cat] += 1

val_ann = [0,0,0,0,0,0,0]
for i in ann_val:
  cat = i['category_id']
  val_ann[cat] += 1

print(train_ann)
print(val_ann)

for i in range(7):
  print(val_ann[i] / train_ann[i])
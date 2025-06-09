import os
import json
import numpy as np

def toStr(annArr):
  str_ann = ""
  for i in annArr:
    str_ann += str(i)
  return str_ann

def toUniqueStr(annArr):
  str_ann = ""
  unique = list(set(annArr))
  unique.sort()
  for i in unique:
    str_ann += str(i)
  return str_ann

def count(patterns):
  classes = [0,0,0,0,0,0,0]
  for k, v in patterns.items():
    for i in k:
      num = int(i)
      classes[num] += v
  return classes

ids = {}
with open('/home/feyad/code/bitirme/datasets/spinexr/train_full_10.json') as f:
    data = json.load(f)
    anns = data["annotations"]
    for ann in anns:
      img_id = ann["image_id"]
      cat_id = ann["category_id"]
      if ids.get(img_id) is None:
        ids[img_id] = []
      ids[img_id].append(cat_id)


sample_anns = []
sample_images = []
sample_categories = []
sample_images_ids = []

patterns = {}
for id, anns in ids.items():
  anns.sort()
  str_ann = toStr(anns)

  if patterns.get(str_ann) is None:
    patterns[str_ann] = 0
  patterns[str_ann] += 1

  if ("0" != toUniqueStr(str_ann)):
    sample_images_ids.append(id)

with open('/home/feyad/code/bitirme/datasets/spinexr/train_full_10.json') as f:
  data = json.load(f)
  anns = data["annotations"]
  imgs = data["images"]
  cats = data["categories"]

  for ann in anns:
    if ann['image_id'] in sample_images_ids:
      sample_anns.append(ann)
  
  for img in imgs:
    if img['id'] in sample_images_ids:
      sample_images.append(img)
  
  sample_categories = cats

ann_c = 0
for ann in sample_anns:
  ann_c += 1
  ann['id'] = ann_c

json_data = {
  "images": sample_images,
  "annotations": sample_anns,
  "categories": sample_categories,
}

print(len(sample_images))
exit()

with open("./datasets/spinexr/train_sample.json", "w") as f:
    json.dump(json_data, f)


filtered_patterns = patterns
# filtered_patterns = {k: v for (k, v) in patterns.items() if "0" not in k}
filtered_patterns = {k: v for (k, v) in patterns.items() if "0" != toUniqueStr(k)}
# filtered_patterns = {k: v for (k, v) in patterns.items() if "0" != toUniqueStr(k) and len(toUniqueStr(k)) == 1}
classes = count(filtered_patterns)
# print(json.dumps(filtered_patterns, indent=1))
print(json.dumps(classes, indent=1))
exit()

uniquePatterns = {}
for id, anns in ids.items():
  anns.sort()
  str_ann = toUniqueStr(anns)

  if uniquePatterns.get(str_ann) is None:
    uniquePatterns[str_ann] = 0
  uniquePatterns[str_ann] += 1


print(json.dumps(uniquePatterns, indent=1))


print(len(list(uniquePatterns.keys())))
print(len(set(list(uniquePatterns.keys()))))

print([f"{k}: {v}" for k,v in uniquePatterns.items() if len(k) == 1])
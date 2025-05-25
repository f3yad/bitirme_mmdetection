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

ids = {}
with open('/home/feyad/code/bitirme/datasets/spinexr/train.json') as f:
    data = json.load(f)
    anns = data["annotations"]
    for ann in anns:
      img_id = ann["image_id"]
      cat_id = ann["category_id"]
      if ids.get(img_id) is None:
        ids[img_id] = []
      ids[img_id].append(cat_id)

patterns = {}
for id, anns in ids.items():
  anns.sort()
  str_ann = toStr(anns)

  if patterns.get(str_ann) is None:
    patterns[str_ann] = 0
  patterns[str_ann] += 1


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
import os
import json
import numpy as np
from collections import defaultdict

to_augment = defaultdict(int)

def ann_arr_to_str(annArr):
  str_ann = ""
  for i in annArr:
    str_ann += str(i)
  return str_ann

def ann_arr_to_unique_str(annArr):
  str_ann = ""
  unique = list(set(annArr))
  unique.sort()
  for i in unique:
    str_ann += str(i)
  return str_ann

imgIdAnnotation = {}
with open('/home/feyad/code/bitirme/datasets/spinexr/train.json') as f:
    data = json.load(f)
    anns = data["annotations"]
    for ann in anns:
      img_id = ann["image_id"]
      cat_id = ann["category_id"]
      if imgIdAnnotation.get(img_id) is None:
        imgIdAnnotation[img_id] = []
      imgIdAnnotation[img_id].append(cat_id)


# example:
# "0012" => 23
patterns = {}
for id, anns in imgIdAnnotation.items():
  anns.sort()
  str_ann = ann_arr_to_str(anns)

  if patterns.get(str_ann) is None:
    patterns[str_ann] = 0
  patterns[str_ann] += 1



uniquePatterns = {}
for id, anns in imgIdAnnotation.items():
  anns.sort()
  str_ann = ann_arr_to_unique_str(anns)

  if uniquePatterns.get(str_ann) is None:
    uniquePatterns[str_ann] = 0
  uniquePatterns[str_ann] += 1

# print(json.dumps(patterns, indent=1))
print("num of images with single anns:")
print([f"{k}: {v}" for k,v in uniquePatterns.items() if len(k) == 1])


count_of_single_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  str_ann = ann_arr_to_unique_str(anns)
  if len(str_ann) == 1:
    count_of_single_ann[str_ann] += len(anns)

print("files with single anns:")
print(count_of_single_ann)



count_of_non_zero_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns:
    for an in anns:
      count_of_non_zero_ann[str(an)] += 1

print("files with non-zero anns:")
print(count_of_non_zero_ann)


count_of_less_than_3_zero_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if anns.count(0) < 7 and ann_arr_to_unique_str(anns) != '0':
    to_augment[id] += 1
    for an in anns:
      count_of_less_than_3_zero_ann[str(an)] += 1

print(" ## files with less than 3 zeros anns:")
print(count_of_less_than_3_zero_ann)


count_of_no_zero_class_6_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 6 in anns:
    to_augment[id] += 15
    for an in anns:
      count_of_no_zero_class_6_ann[str(an)] += 1

print("## files with only non-zero class-6 anns:")
print(count_of_no_zero_class_6_ann)


count_of_no_zero_class_5_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 5 in anns:
    to_augment[id] += 2
    for an in anns:
      count_of_no_zero_class_5_ann[str(an)] += 1

print("## files with only non-zero class-5 anns:")
print(count_of_no_zero_class_5_ann)


count_of_no_zero_class_4_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 4 in anns:
    to_augment[id] += 5
    for an in anns:
      count_of_no_zero_class_4_ann[str(an)] += 1

print("## files with only non-zero class-4 anns:")
print(count_of_no_zero_class_4_ann)


count_of_no_zero_class_3_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 3 in anns:
    to_augment[id] += 2
    for an in anns:
      count_of_no_zero_class_3_ann[str(an)] += 1

print("## files with only non-zero class-3 anns:")
print(count_of_no_zero_class_3_ann)


count_of_no_zero_class_2_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 2 in anns:
    to_augment[id] += 1
    for an in anns:
      count_of_no_zero_class_2_ann[str(an)] += 1

print("## files with only non-zero class-2 anns:")
print(count_of_no_zero_class_2_ann)


count_of_no_zero_class_1_ann = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for id, anns in imgIdAnnotation.items():
  anns.sort()
  if 0 not in anns and 1 in anns:
    to_augment[id] += 22
    for an in anns:
      count_of_no_zero_class_1_ann[str(an)] += 1

print("## files with only non-zero class-1 anns:")
print(count_of_no_zero_class_1_ann)




aug = [
  count_of_less_than_3_zero_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,
  count_of_no_zero_class_6_ann,

  count_of_no_zero_class_5_ann,
  count_of_no_zero_class_5_ann,
  
  count_of_no_zero_class_4_ann,
  count_of_no_zero_class_4_ann,
  count_of_no_zero_class_4_ann,
  count_of_no_zero_class_4_ann,
  count_of_no_zero_class_4_ann,
  
  count_of_no_zero_class_3_ann,
  count_of_no_zero_class_3_ann,
  
  # count_of_no_zero_class_2_ann,
  
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
  count_of_no_zero_class_1_ann,
]




sum_anns = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

for a in aug:
  for k, v in a.items():
    sum_anns[k] += v

print("SUM:")
print(sum_anns)

print(len(to_augment.keys()))
print(sum(list(to_augment.values())))

# with open("./sample.json", 'w') as f:
#     json.dump(to_augment, f)


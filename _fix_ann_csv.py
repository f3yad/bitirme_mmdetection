import pandas as pd
import json
import os

df = pd.read_csv('/home/feyad/code/bitirme/datasets/spinexr-new/_________________veri_kumesini_bolme/train___split.csv', sep=';')

categories = {
"Osteophytes": 0,
"Spondylolysthesis": 1,
"Disc space narrowing": 2,
"Other lesions": 3,
"Surgical implant": 4,
"Foraminal stenosis": 5,
"Vertebral collapse": 6,
}

categories = {
  0: "Osteophytes",
  1: "Spondylolysthesis",
  2: "Disc space narrowing",
  3: "Other lesions",
  4: "Surgical implant",
  5: "Foraminal stenosis",
  6: "Vertebral collapse",
}

ids = []

file = "study_id;series_id;image_id;rad_id;lesion_type;xmin;ymin;xmax;ymax\n"
with open('/home/feyad/code/bitirme/datasets/spinexr/train_v2.json') as f:
  data = json.load(f)
  count = 0
  for img in data['annotations']:
    row = f"{img['image_id']};{img['image_id']};{img['image_id']};{img['image_id']};{categories[img['category_id']]};{img['bbox'][0]};{img['bbox'][1]};{img['bbox'][0]+img['bbox'][2]};{img['bbox'][1]+img['bbox'][3]}\n"
    file += row
    count += 1

  c= 0
  for index, row in df.iterrows():
    if row['lesion_type'] == "No finding":
      c +=1
      x = f'{df.loc[[index]].to_csv(index=False, header=False, sep=";")}'
      file += x

with open("/home/feyad/code/bitirme/datasets/spinexr-new/_________________veri_kumesini_bolme/train_split_v2.csv", "w") as f:
    f.write(file)
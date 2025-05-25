import albumentations as A
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw

target_bbox_counts = [
  7000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
]

# imgid = "db891d3f337735034fc12d76fefa3397"
imgid = "774cadf3911065ffccc8dbcbc47bd1e8"
imgpath = f"./{imgid}.png"

anns = []
with open('/home/feyad/code/bitirme/datasets/spinexr/val.json') as f:
    data = json.load(f)
    for ann in data['annotations']:
      if ann['image_id'] == imgid:
        anns.append(ann)

    print(json.dumps(anns, indent=2))
   

colors = ["pink", "white", "red", "blue", "orange", "yellow", "darkgreen"]


def augmentation(imgpath, img_anns):
  bboxes = []
  labels = []
  for ann in anns:
      bboxes.append(ann['bbox'])
      labels.append((ann['category_id']))

  # Load or create an image (NumPy array)
  image = cv2.imread(imgpath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # 1. Instantiate the transform
  pipeline = A.Compose([
    A.HorizontalFlip(p=1.0) # p=1.0 means always apply
  ],
    bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
  )

  # 2. Apply the transform to the image
  augmented = pipeline(image=image, bboxes=bboxes, labels=labels)
  transformed_image = augmented['image']
  transformed_bboxes = augmented['bboxes']
  transformed_labels = [int(x) for x in augmented['labels']]
  print(labels)
  print(transformed_labels)
  outpath = f"{imgpath}_aug.png"
  cv2.imwrite(outpath, transformed_image)
  print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
  draw(imgpath, bboxes, labels)
  draw(outpath, transformed_bboxes, transformed_labels)

def draw(imgpath, bboxes, labels):
  image = Image.open(imgpath).convert("RGB")
  draw = ImageDraw.Draw(image)
  for bbox, label in zip(bboxes, labels):
    x, y, w, h = bbox
    draw.rectangle([x, y, x+w, y+h], outline=colors[label], width=3)
  image.save(f"{imgpath}_draw.png")
  print(f"Saved image with bounding boxes")

augmentation(imgpath, anns)
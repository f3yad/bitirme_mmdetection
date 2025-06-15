import albumentations as A
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
import uuid
import os

img_counts = 0

colors = ["pink", "white", "red", "blue", "orange", "yellow", "darkgreen"]

to_augment = None
with open('./sample.json') as f:
    to_augment = json.load(f)

images = {}
with open('./datasets/spinexr/train.json') as f:
    data = json.load(f)

    for img in data["images"]:
      imgid = img["id"]
      
      if images.get(imgid) is None and imgid in to_augment:
        images[imgid] = {"id": imgid,"w": img["width"], "h": img["height"], "anns": []}

    for ann in data['annotations']:
      imgid = ann['image_id']
      if images.get(imgid) is None:
        print(f"ERROR: ANN: {imgid}")
      else:
        images[imgid]["anns"].append(ann)

def augmentation(imgpath, img_anns):
  bboxes = []
  labels = []
  for ann in img_anns:
      bboxes.append(ann['bbox'])
      labels.append((ann['category_id']))

  # Load or create an image (NumPy array)
  image = cv2.imread(imgpath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # 1. Instantiate the transform
  pipeline = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.2),  # swapping rows X columns
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(-0.05,0.2), p=0.5),
    A.BBoxSafeRandomCrop(p=0.5),
    A.Rotate(limit=(-90, 90) , p=0.5),

    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(-0.05,0.2), p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    A.BBoxSafeRandomCrop(p=0.5),
    A.Affine(scale=[0.8, 1.2],translate_percent=[-0.1, 0.1],rotate=[-4, 4],shear="0",interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST,fit_output=False,keep_ratio=False,rotate_method="ellipse",balanced_scale=True,border_mode=cv2.BORDER_CONSTANT,fill=0,fill_mask=0)
  ],
    bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
  )

  # 2. Apply the transform to the image
  augmented = pipeline(image=image, bboxes=bboxes, labels=labels)
  transformed_image = augmented['image']
  transformed_bboxes = augmented['bboxes']
  transformed_labels = [int(x) for x in augmented['labels']]
  return (transformed_image, transformed_bboxes, transformed_labels, bboxes, labels)

def draw(imgpath, bboxes, labels, outpath):
  image = Image.open(imgpath).convert("RGB")
  draw = ImageDraw.Draw(image)
  for bbox, label in zip(bboxes, labels):
    x, y, w, h = bbox
    draw.rectangle([x, y, x+w, y+h], outline=colors[label], width=3)
  image.save(outpath)
  print(f"Saved image with bounding boxes in {outpath}")


train_images_dir = "./datasets/spinexr/train_images"
out_train_images_dir = "./datasets/spinexr/train_images_sample"
draw_out_train_images_dir = "./datasets/spinexr/train_images_aug_draw"
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(out_train_images_dir, exist_ok=True)
os.makedirs(draw_out_train_images_dir, exist_ok=True)

anns_text = "image_id,category_id,xmin,ymin,w,h\n"

counter = 0
for ___, img in images.items():
  imgpath = f"{train_images_dir}/{img['id']}.png"
  # print(img["id"], to_augment[img["id"]])
  count = to_augment[img["id"]]
  for _ in range(count):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}]"
    print(f"{counter_msg} :: original labels: {labels}")
    print(f"{counter_msg} :: augmented labels: {t_labels}")

    outpath = f"{out_train_images_dir}/{img['id']}_{random_uuid}.png"
    cv2.imwrite(outpath, t_image)
    img_counts += 1
    # print(f"{counter_msg} :: Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")

    org_draw_path = f"{draw_out_train_images_dir}/{img['id']}_{random_uuid}_org_draw.png"
    aug_draw_path = f"{draw_out_train_images_dir}/{img['id']}_{random_uuid}_aug_draw.png"
    # draw(imgpath, bboxes, labels, org_draw_path)
    # draw(outpath, t_bboxes, t_labels, aug_draw_path)

print(f'IMG COUNTS = {img_counts}')
with open("./datasets/spinexr/train_aug_sample.csv", "w") as f:
    f.write(anns_text)
# augmentation(sample_imgpath, anns)
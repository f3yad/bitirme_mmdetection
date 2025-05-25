import albumentations as A
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
import uuid
import os

img_counts = 0

colors = ["pink", "white", "red", "blue", "orange", "yellow", "darkgreen"]

images = {}
with open('/home/feyad/code/bitirme/datasets/spinexr/train.json') as f:
    data = json.load(f)

    for img in data["images"]:
      imgid = img["id"]
      if images.get(imgid) is None:
        images[imgid] = {"id": imgid,"w": img["width"], "h": img["height"], "anns": []}

    for ann in data['annotations']:
      imgid = ann['image_id']
      if images.get(imgid) is None:
        print(f"ERROR: ANN: {imgid}")
      else:
        images[imgid]["anns"].append(ann)

    only_0 = 0
    shared_0 = 0
    other = 0
    no_zero_anns_images = []
    for image_id, image_ann in images.items():
      cat_ids = []
      for ann_bbox in image_ann['anns']:
        cat_id = ann_bbox['category_id']
        if cat_id not in cat_ids: cat_ids.append(cat_id)
      
      if len(cat_ids) == 1 and cat_ids[0] == 0:
        only_0 += 1
      elif len(cat_ids) > 1 and 0 in cat_ids:
        shared_0 += 1
      else:
        other += 1
        no_zero_anns_images.append(image_ann)
        if len(cat_ids) > 0: print(cat_ids)
    
    nonempty_anns_images_no_zero = [ann for ann in no_zero_anns_images if len(ann['anns']) > 0]
    print(json.dumps(nonempty_anns_images_no_zero, indent=2))
    print(f"0: {only_0}")
    print(f"+0: {shared_0}")
    print(f"-0: {other}")
    print(len(data['annotations']))
print(len(nonempty_anns_images_no_zero))



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
out_train_images_dir = "./datasets/spinexr/train_images_aug"
draw_out_train_images_dir = "./datasets/spinexr/train_images_aug_draw"
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(out_train_images_dir, exist_ok=True)
os.makedirs(draw_out_train_images_dir, exist_ok=True)

anns_text = "image_id,category_id,xmin,ymin,w,h\n"

counter = 0
for img in nonempty_anns_images_no_zero:
  imgpath = f"{train_images_dir}/{img['id']}.png"
  
  for _ in range(16):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}/{len(nonempty_anns_images_no_zero)*16}]"
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

counter = 0
custom = [ann for ann in nonempty_anns_images_no_zero if len(ann["anns"]) == 1 and ann["anns"][0]["category_id"] in [1, 6] ]
# custom = [ann for ann in nonempty_anns_images_no_zero if len(ann["anns"]) != 1 ]

for img in custom:
  imgpath = f"{train_images_dir}/{img['id']}.png"
  for _ in range(20):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}/{len(custom)*20}]"
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

# counter = 0
# custom = [ann for ann in nonempty_anns_images_no_zero if 5 not in ann["anns"] ]

# def custom_filter(item):
#   for i in item['anns']:
#     if i['category_id'] == 5:
#       return False
#   return True

# custom = list(filter(custom_filter, custom))

# for img in custom:
#   imgpath = f"{train_images_dir}/{img['id']}.png"
#   for _ in range(15):
#     random_uuid = uuid.uuid4()
#     t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

#     for bbox, label in zip(t_bboxes, t_labels):
#       ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
#       anns_text += ann_text

#     counter += 1
#     counter_msg = f"[{counter}/{len(custom)*15}]"
#     print(f"{counter_msg} :: original labels: {labels}")
#     print(f"{counter_msg} :: augmented labels: {t_labels}")

#     outpath = f"{out_train_images_dir}/{img['id']}_{random_uuid}.png"
#     cv2.imwrite(outpath, t_image)
#     img_counts += 1
#     # print(f"{counter_msg} :: Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")

#     org_draw_path = f"{draw_out_train_images_dir}/{img['id']}_{random_uuid}_org_draw.png"
#     aug_draw_path = f"{draw_out_train_images_dir}/{img['id']}_{random_uuid}_aug_draw.png"
#     # draw(imgpath, bboxes, labels, org_draw_path)
#     # draw(outpath, t_bboxes, t_labels, aug_draw_path)

counter = 0
custom = [ann for ann in nonempty_anns_images_no_zero if (len(ann["anns"]) == 1 and ann["anns"][0]["category_id"] in [1, 6]) or (len(ann["anns"]) == 2 and f'{ann["anns"][0]["category_id"]}{ann["anns"][1]["category_id"]}' in ["16", "61"]) ]

for img in custom:
  imgpath = f"{train_images_dir}/{img['id']}.png"
  for _ in range(250):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}/{len(custom)*250}]"
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


counter = 0
custom = [ann for ann in nonempty_anns_images_no_zero if len(ann["anns"]) == 1 and ann["anns"][0]["category_id"] == 6]

for img in custom:
  imgpath = f"{train_images_dir}/{img['id']}.png"
  for _ in range(250):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}/{len(custom)*250}]"
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


counter = 0
custom = [ann for ann in nonempty_anns_images_no_zero if len(ann["anns"]) == 1 and ann["anns"][0]["category_id"] in [2,3,4]]

for img in custom:
  imgpath = f"{train_images_dir}/{img['id']}.png"
  for _ in range(50):
    random_uuid = uuid.uuid4()
    t_image, t_bboxes, t_labels, bboxes, labels = augmentation(imgpath, img['anns'])

    for bbox, label in zip(t_bboxes, t_labels):
      ann_text = f"{img['id']}_{random_uuid},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n"
      anns_text += ann_text

    counter += 1
    counter_msg = f"[{counter}/{len(custom)*50}]"
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
with open("./datasets/spinexr/train_aug_10.csv", "w") as f:
    f.write(anns_text)
# augmentation(sample_imgpath, anns)
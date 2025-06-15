import json
from collections import defaultdict
import os

### TRAIN
# json_ann_path = "./datasets/spinexr/train_fixed.json"
# images_dir_path = "./datasets/spinexr/train_images"
# output_labels_path = "./datasets/spinexr/yolodata/train/labels"

### VAL
json_ann_path = "./datasets/spinexr/val_fixed.json"
images_dir_path = "./datasets/spinexr/val_images"
output_labels_path = "./datasets/spinexr/yolodata/val/labels"

### TEST
# json_ann_path = "./datasets/spinexr/test_fixed.json"
# images_dir_path = "./datasets/spinexr/test_images"
# output_labels_path = "./datasets/spinexr/yolodata/test/labels"

os.makedirs(output_labels_path, exist_ok=True)

with open(json_ann_path, "r") as f:
    coco = json.load(f)

# Map category id to name
image_id_to_ann_arr = {}
for img in coco["images"]:
  img_id = img["id"]
  if not image_id_to_ann_arr.get(img_id):
    image_id_to_ann_arr[img_id] = {
      "anns": [],
      "w": img["width"],
      "h": img["height"],
    }


for ann in coco["annotations"]:
  img_id = ann["image_id"]
  image_id_to_ann_arr[img_id]["anns"].append(ann)

no_anns_images_count = 0
anns_images_count = 0
for img_id, img in image_id_to_ann_arr.items():
  txt_file_name = f"{output_labels_path}/{img_id}.txt"

  if len(img["anns"]) == 0:
    no_anns_images_count += 1
    open(txt_file_name, "w").close()
    print(f"[created]: {txt_file_name} - EMPTY")
  else:
    anns_images_count += 1

    lines = []
    for ann in img["anns"]:
      min_x, min_y, w, h = ann['bbox']
      cn = ann['category_id']
      img_w = img["w"]
      img_h = img["h"]

      x_center = min_x + w / 2
      y_center = min_y + h / 2

      norm_x_cntr = x_center / img_w
      norm_y_cntr = y_center / img_h
      norm_w = w / img_w
      norm_h = h / img_h

      line = f"{cn} {norm_x_cntr} {norm_y_cntr} {norm_w} {norm_h}"
      lines.append(line)

    with open(txt_file_name, "w") as f:
      for line in lines:
        f.write(line + "\n")

    print(f"[created]: {txt_file_name} - {len(lines)} LINE OF ANNOTATIONS")


print(f"no_anns_images_count: {no_anns_images_count}")
print(f"anns_images_count: {anns_images_count}")
print(f"sum: {no_anns_images_count + anns_images_count}")


exit()
# Map image id to filename
image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

# Group annotations per image
annotations_per_image = defaultdict(list)
for ann in coco["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    bbox = ann["bbox"]  # [x, y, width, height]

    # Convert bbox to YOLO format: [x_center, y_center, width, height], normalized
    image_info = next(img for img in coco["images"] if img["id"] == image_id)
    img_width = image_info["width"]
    img_height = image_info["height"]

    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height

    yolo_bbox = [category_id, x_center, y_center, w, h]
    annotations_per_image[image_id].append(yolo_bbox)

# Example: print annotations for each image
for image_id, anns in annotations_per_image.items():
    print(f"Image: {image_id_to_filename[image_id]}")
    for ann in anns:
        class_id, x_center, y_center, w, h = ann
        class_name = category_id_to_name[class_id]
        print(f"  Class: {class_name} ({class_id}), BBox: {x_center:.3f}, {y_center:.3f}, {w:.3f}, {h:.3f}")

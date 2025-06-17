# from ensemble_boxes import soft_nms
# import json

# from collections import defaultdict

# def load_predictions(file_path):
#     with open(file_path, 'r') as f:
#         raw = json.load(f)
#     grouped = defaultdict(list)
#     for item in raw:
#         image_id = item['image_id']
#         grouped[image_id].append(item)
#     return grouped

# def normalize_bbox(bbox, w, h):
#     x, y, bw, bh = bbox
#     return [
#         x / w,
#         y / h,
#         (x+bw) / w,
#         (y+bh) / h
#     ]

# def fuse_ensemble(image_ids, yolov8_preds, faster_preds):
#     fused_results = []
#     for image_id in image_ids:
#         boxes_list = []
#         scores_list = []
#         labels_list = []

#         for preds in [yolov8_preds, faster_preds]:
#             for pred in preds.get(image_id, []):
#                 bbox = normalize_bbox(pred['bbox'], 1024, 1024)
#                 boxes_list.append(bbox)
#                 scores_list.append(pred['score'])
#                 labels_list.append(pred['category_id'])

#         boxes, scores, labels = soft_nms([boxes_list], 
#                                          [scores_list], 
#                                          [labels_list], 
#                                         #  iou_thr=0.5,
#                                          thresh=0.3,
#                                          sigma=0.5, 
#                                         #  method='linear'
#                                          )
        
#         for box, score, label in zip(boxes, scores, labels):
#             x_min = box[0] * 1024
#             y_min = box[1] * 1024
#             x_max = box[2] * 1024
#             y_max = box[3] * 1024
#             fused_results.append({
#                 'image_id': image_id,
#                 'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
#                 'category_id': int(label),
#                 'score': score
#             })
        
#     return fused_results

# yolo_preds = load_predictions("f_yolo_predictions.json")
# faster_preds = load_predictions("results/faster.bbox.json")
# image_ids = list(set(yolo_preds.keys()) | set(faster_preds.keys()))
# fused = fuse_ensemble(image_ids, yolo_preds, faster_preds)
# # Save the fused results to a JSON file
# with open("f_ensembling_results.json", "w") as f:
#     json.dump(fused, f, indent=4)





# from ensemble_boxes import soft_nms
# import json
# from collections import defaultdict

# def load_preds(json_path):
#     with open(json_path) as f:
#         raw = json.load(f)
#     grouped = defaultdict(list)
#     for pred in raw:
#         grouped[pred['image_id']].append(pred)
#     return grouped

# def normalize_bbox(bbox, w, h):
#     print(bbox)
#     x, y, bw, bh = bbox
#     return [x / w, y / h, (x + bw) / w, (y + bh) / h]

# def fuse_ensemble(image_ids, yolov8_preds, faster_preds):
#     fused_results = []
#     for img_id in image_ids:
#         boxes_list, scores_list, labels_list = [], [], []

#         for preds in [yolov8_preds, faster_preds]:
#             for pred in preds.get(img_id, []):
#                 if pred['score'] <= 0:
#                     continue

#                 bbox = normalize_bbox(pred['bbox'], 640, 640)
#                 x1, y1, x2, y2 = bbox
#                 if x2 <= x1 or y2 <= y1:
#                     continue

#                 boxes_list.append(bbox)
#                 scores_list.append(pred['score'])
#                 labels_list.append(pred['category_id'])

#         if len(boxes_list) == 0:
#             print(f"[Warning] No valid boxes for image_id {img_id}, skipping.")
#             continue

#         boxes, scores, labels = soft_nms(
#             [boxes_list], [scores_list], [labels_list],
#             sigma=0.5, thresh=0.3
#         )


#         for box, score, label in zip(boxes, scores, labels):
#             x_min = box[0] * 1024
#             y_min = box[1] * 1024
#             x_max = box[2] * 1024
#             y_max = box[3] * 1024
#             fused_results.append({
#                 "image_id": img_id,
#                 "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
#                 "score": score,
#                 "category_id": int(label)
#             })

#     return fused_results

# # Load predictions
# yolo_preds = load_preds('f_yolo_predictions.json')
# faster_preds = load_preds('results/faster.bbox.json')

# # Union of all image IDs from both models
# image_ids = list(set(yolo_preds.keys()) | set(faster_preds.keys()))

# # Run fusion
# fused = fuse_ensemble(image_ids, yolo_preds, faster_preds)

# # Save fused results
# with open('fused_softnms_results.json', 'w') as f:
#     json.dump(fused, f)



from ensemble_boxes import soft_nms
from collections import defaultdict
import json
import os

json_path = "datasets/spinexr/test_fixed.json"
with open(json_path, 'r') as f:
    data = json.load(f)

image_sizes = {}
for img in data['images']:
    img_id = img['id']
    width = img['width']
    height = img['height']
    image_sizes[img_id] = [width, height]


#################################################


def load_preds(json_path, model_name):
    with open(json_path) as f:
        raw = json.load(f)
    grouped = defaultdict(list)
    for pred in raw:
        img_id = pred['image_id']
        if model_name == 'yolo':
            img_id = img_id.replace("\\", "/")[2:]
            filename = os.path.splitext(img_id)[0]
            filename = filename.split("/")[-1]
            img_id = os.path.splitext(filename)[0]
        grouped[img_id].append(pred)
    return grouped

def normalize_bbox(bbox, w, h, img_id):
    x, y, bw, bh = bbox
    n_min_x = x / w
    n_min_y = y / h
    n_max_x = (x + bw) / w
    n_max_y = (y + bh) / h

    if n_min_x > 1 or n_min_y > 1 or n_max_x > 1 or n_max_y > 1:
        print(bbox, img_id, w, h, [n_min_x, n_min_y, n_max_x, n_max_y])
    return [n_min_x, n_min_y, n_max_x, n_max_y]

def fuse_ensemble(image_ids, yolov8_preds, faster_preds):
    fused_results = []
    for img_id in image_ids:
        filename = os.path.basename(img_id)
        boxes_all_models = []
        scores_all_models = []
        labels_all_models = []

        for preds in [yolov8_preds, faster_preds]:
            boxes, scores, labels = [], [], []

            for pred in preds.get(img_id, []):
                if pred['score'] <= 0:
                    continue

                bbox = normalize_bbox(pred['bbox'], image_sizes[img_id][0], image_sizes[img_id][1], img_id)
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append(bbox)
                scores.append(pred['score'])
                labels.append(pred['category_id'])

            if boxes:
                boxes_all_models.append(boxes)
                scores_all_models.append(scores)
                labels_all_models.append(labels)

        if len(boxes_all_models) == 0:
            print(f"[Warning] No valid boxes for image_id {img_id}, skipping.")
            continue

        boxes, scores, labels = soft_nms(
            boxes_all_models,
            scores_all_models,
            labels_all_models,
            sigma=0.5,
            thresh=0.3
        )

        for box, score, label in zip(boxes, scores, labels):
            x_min = box[0] * image_sizes[img_id][0]
            y_min = box[1] * image_sizes[img_id][1]
            x_max = box[2] * image_sizes[img_id][0]
            y_max = box[3] * image_sizes[img_id][1]
            fused_results.append({
                "image_id": img_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "score": score,
                "category_id": int(label)
            })

    return fused_results

# Load predictions
yolo_preds = load_preds('f_yolo_predictions.json', "yolo")
faster_preds = load_preds('results/faster.bbox.json', "faster")

print(len(yolo_preds))
print(len(faster_preds))

# Union of all image IDs from both models
image_ids = list(set(yolo_preds.keys()) | set(faster_preds.keys()))

# Run fusion
fused = fuse_ensemble(image_ids, yolo_preds, faster_preds)

# Save fused results
with open('fused_softnms_results.json', 'w') as f:
    json.dump(fused, f)

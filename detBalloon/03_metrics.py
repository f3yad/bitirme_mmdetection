import mmcv
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmdet.registry import DATASETS
from mmengine.registry import init_default_scope
from mmengine.fileio import load
from mmdet.evaluation.functional import bbox_overlaps

# --- Load config and dataset ---
config_file = 'configs/rtmdet_tiny_1xb4-20e_balloon.py'  # change this
pkl_file = 'detBalloon/balloon.pkl'  # change if needed

cfg = Config.fromfile(config_file)
init_default_scope(cfg.get('default_scope', 'mmdet'))

cfg.test_dataloader.dataset.test_mode = True
dataset = DATASETS.build(cfg.test_dataloader.dataset)


# --- Load predictions ---
# results = mmcv.load(pkl_file)
results = load(pkl_file)

# Set IoU threshold to count a match
iou_threshold = 0.5

TP, FP, FN = 0, 0, 0

for idx in tqdm(range(len(dataset))):
    data_info = dataset.get_data_info(idx)
    instances = data_info.get('instances', [])
    gt_bboxes = np.array([ins['bbox'] for ins in instances])
    pred_instances = results[idx]['pred_instances']
    pred_bboxes = pred_instances['bboxes'].cpu().numpy()
    if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
        continue  # nothing to match, skip
    elif len(pred_bboxes) == 0:
        FN += len(gt_bboxes)
        continue
    elif len(gt_bboxes) == 0:
        FP += len(pred_bboxes)
        continue

    ious = bbox_overlaps(pred_bboxes, gt_bboxes)
    matched_gt = set()
    for i in range(len(pred_bboxes)):
        max_iou = 0
        match_j = -1
        for j in range(len(gt_bboxes)):
            if j in matched_gt:
                continue
            iou = ious[i, j]
            if iou > max_iou:
                max_iou = iou
                match_j = j
        if max_iou >= iou_threshold:
            TP += 1
            matched_gt.add(match_j)
        else:
            FP += 1
    FN += len(gt_bboxes) - len(matched_gt)

# --- Calculate precision and recall ---
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

print(f"True Precision: {precision:.4f}")
print(f"True Recall:    {recall:.4f}")

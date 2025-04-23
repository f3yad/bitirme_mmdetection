# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls

@METRICS.register_module()
class YOLOStylePR(BaseMetric):
    def __init__(self, iou_threshold=0.5, collect_device="cpu", prefix=None, **kwargs):
        # super().__init__(**kwargs)
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_threshold = iou_threshold

    def process(self, data_batch, data_samples):
        """Collect predictions and ground truths for evaluation."""
        for data in data_samples:
            pred_instances = data['pred_instances']
            gt_instances = data['gt_instances']
            self.results.append((pred_instances, gt_instances))

    def compute_metrics(self, results):
        """Compute P/R for each class."""
        logger = MMLogger.get_current_instance()
        classwise = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

        for preds, gts in results:
            pred_bboxes = preds['bboxes'].cpu().numpy()
            pred_labels = preds['labels'].cpu().numpy()
            gt_bboxes = gts['bboxes'].cpu().numpy()
            gt_labels = gts['labels'].cpu().numpy()

            matched_gt = set()
            for pb, pl in zip(pred_bboxes, pred_labels):
                ious = bbox_iou(pb, gt_bboxes)
                match_idx = np.argmax(ious)
                if ious[match_idx] >= self.iou_threshold and gt_labels[match_idx] == pl and match_idx not in matched_gt:
                    classwise[pl]['TP'] += 1
                    matched_gt.add(match_idx)
                else:
                    classwise[pl]['FP'] += 1

            # Count FN (missed gts)
            for i, gl in enumerate(gt_labels):
                if i not in matched_gt:
                    classwise[gl]['FN'] += 1

        metrics = {}
        for cls, stats in classwise.items():
            TP = stats['TP']
            FP = stats['FP']
            FN = stats['FN']
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            metrics[f'class_{cls}_P'] = prec
            metrics[f'class_{cls}_R'] = rec

        return metrics


def bbox_iou(box, boxes):
    """IoU between a single box and multiple boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / (union_area + 1e-6)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO('datasets/spinexr/test_fixed.json')
dt = gt.loadRes('./fused_softnms_results.json')
eval = COCOeval(gt, dt, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()

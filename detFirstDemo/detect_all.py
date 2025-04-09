import torch
import mmengine.logging.history_buffer
import numpy.core.multiarray
from mmdet.apis import DetInferencer
import os

# Choose to use a config
model_name = 'rtmdet_tiny_8xb32-300e_coco'

# Setup a checkpoint file to load
checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(model_name, checkpoint, device)

# Use the detector to do inference
img = './datasets/firstDemo/elazig_iki.png'
result = inferencer(img, out_dir='./output/firstDemo')

dir_name = "./datasets/firstDemo/"
all_files = os.listdir(dir_name)

for f in all_files:
  img = dir_name + f
  print(img)
  result = inferencer(img, out_dir='./output/firstDemo')

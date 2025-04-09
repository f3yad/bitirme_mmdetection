from mmdet.apis import DetInferencer
import glob
import os

# Choose to use a config
config = 'configs/rtmdet_tiny_1xb4-20e_balloon.py'
# Setup a checkpoint file to load
checkpoint = glob.glob('./work_dirs/rtmdet_tiny_1xb4-20e_balloon/best_coco*.pth')[0]

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

dir_name = "./datasets/balloon/val/"
all_files = os.listdir(dir_name)
jpg_files = list(filter(lambda f: f.endswith(".jpg") ,all_files))

for f in jpg_files:
  img = dir_name + f
  print(img)
  result = inferencer(img, out_dir='./output/balloon')


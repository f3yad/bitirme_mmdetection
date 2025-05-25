from mmdet.apis import DetInferencer
import glob

# Choose to use a config
config = 'configs/rtmdet_tiny_1xb4-20e_spinexr_with-yolo-metrics.py'
# Setup a checkpoint file to load
checkpoint = glob.glob('./work_dirs/rtmdet_tiny_1xb4-20e_spinexr_with-yolo-metrics/best_coco_Osteophytes_precision_epoch_19*.pth')[0]

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

# Use the detector to do inference
img = './datasets/spinexr/train_images_jpg/0a5d7e3f44d6f54a4d65e7b6f06b73a3.jpg'
result = inferencer(img, out_dir='./output/spinexr')
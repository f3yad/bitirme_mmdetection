from PIL import Image
import numpy as np
import os

images_path = "datasets/spinexr/train_images_aug/"
images_names = os.listdir(images_path)

images_count = len(images_names)
count = 0
channels = {
  2: 0,
  3: 0,
}
for img_name in images_names:
  count += 1
  img_path = f'{images_path}{img_name}'  # replace this with a real image path
  img = Image.open(img_path)
  img_np = np.array(img)
  ch = len(img_np.shape)
  channels[ch] += 1
  print(f'[{count}/{images_count}] [{len(img_np.shape)}]: {img_np.shape}')

print(channels)
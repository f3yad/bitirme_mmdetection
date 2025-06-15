# import os
# import shutil

# files = os.listdir("/home/feyad/code/bitirme/datasets/spinexr/train_images")
# files2 = os.listdir("/home/feyad/code/bitirme/datasets/spinexr/yolodata/train/labels")

# imgs = []
# labels = []
# for filename in files:
#     name, ext = os.path.splitext(filename)
#     imgs.append(name)

# for filename in files2:
#     name, ext = os.path.splitext(filename)
#     labels.append(name)

# count = 0
# for i in labels:
#   count += 1
#   print(f"{count} / {len(labels)}")
  
#   shutil.copy(
#     f"/home/feyad/code/bitirme/datasets/spinexr/train_images/{i}.png",
#     f"/home/feyad/code/bitirme/datasets/spinexr/yolodata/train/images/{i}.png"
#   )




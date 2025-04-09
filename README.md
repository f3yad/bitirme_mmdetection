# Object Detection with MMDetection

<details>
  <summary style="font-size: 1.3rem">Prepare Python env with Conda</summary>

  ### create a new conde env with python 3.9:
  ```bash
  conda create -n bitirmeProjesi
  conda install python=3.9
  ```

  ### install pytorch 2.4 with cuda 12.4:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
  ```

  ### verify pytorch installation:
  ```bash
  python _env/verify_pytorch.py.py
  ```

  ### install openmim, mmengine, mmcv and mmdet:
  ```bash
  pip install -U openmim
  mim install mmengine
  CXXFLAGS="-std=c++17" pip install mmcv
  mim install mmdet
  ```

  ### install future and tensorboard:
  ```bash
  pip install future tensorboard
  ```
</details>

<details>
  <summary style="font-size: 1.3rem">Example: Working with balloon dataset</summary>

  ### convert balloon to coco format:
  
  ```bash
  python detBalloon/01_convert_balloon_to_coco_format.py
  ```

  ### create the config file:
  ```bash
  python detBalloon/02_create_config_file.py
  ```

  ### train:
  ```bash
  python mmdetection/tools/train.py configs/rtmdet_tiny_1xb4-20e_balloon.py
  ```

  ### create .pkl file:
  you can also change epoch to the best epoch i.e: best_coco_bbox_mAP_epoch_XX.pth
  ```bash
  python mmdetection/tools/test.py configs/rtmdet_tiny_1xb4-20e_balloon.py work_dirs/rtmdet_tiny_1xb4-20e_balloon/epoch_20.pth --out detBalloon/balloon.pkl
  ```

  ### run detection for single image:
  for single image
  ```bash
  python detBalloon/detect_single.py
  ```
  for all images
  ```bash
  python detBalloon/detect_all.py
  ```
</details>







# Object Detection with MMDetection

<details>
  <summary style="font-size: 1.3rem">Prepare Python env with Conda</summary>

  ### create a new conde env with python 3.9:
  ```bash
  conda create -n bitirmeProjesi
  conda activate bitirmeProjesi
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

  ### install openmim, mmengine, mmcv, mmdet and mmyolo:
  ```bash
  pip install -U openmim
  mim install mmengine
  CXXFLAGS="-std=c++17" pip install mmcv

  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  CXXFLAGS="-std=c++17" pip install -v -e .

  git clone https://github.com/open-mmlab/mmyolo.git
  cd mmyolo
  pip install -r requirements/albu.txt
  CXXFLAGS="-std=c++17" pip install -v -e .
  ```
  ### edit the "mmcv_maximum_version" variable in the following files to be like this:
  * /mmdetection/mmdet/\_\_init\_\_.py
  * /mmyolo/mmyolo/\_\_init\_\_.py
  ```bash
  mmcv_maximum_version = '2.2.1'
  ```

  ### install future and tensorboard:
  ```bash
  pip install future tensorboard
  ```

</details>




<details>
  <summary style="font-size: 1.3rem">Example: First Demo (testing)</summary>

  ### run detection:
  for all images
  ```bash
  python detFirstDemo/detect_all.py
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

  ### prepare the custom evaluator:
  * copy the custom evaluator to the correct location:
  ```bash
  cp _to_copy/yolo_style_pr.py mmdetection/mmdet/evaluation/metrics/
  ```
  * edit the "mmdetection/mmdet/evaluation/metrics/__init__.py" file
  * append "YOLOStylePR" to the \_\_all\_\_ list


  ### train:
  * without yolo metrics
  ```bash
  python -W ignore mmdetection/tools/train.py configs/rtmdet_tiny_1xb4-20e_balloon.py
  ```
  * with yolo metrics (custom evaluator)
  ```bash
  python -W ignore mmdetection/tools/train.py configs/rtmdet_tiny_1xb4-20e_balloon_with-yolo-metrics.py
  ```
  ### create .pkl file:
  you can also change epoch to the best epoch i.e: best_coco_bbox_mAP_epoch_XX.pth
  ```bash
  python mmdetection/tools/test.py configs/rtmdet_tiny_1xb4-20e_balloon.py work_dirs/rtmdet_tiny_1xb4-20e_balloon/epoch_20.pth --out detBalloon/balloon.pkl
  ```

  ### show True Recall and True Precision values:
  ```bash
  python detBalloon/03_metrics.py
  ```

  ### run detection:
  for single image
  ```bash
  python detBalloon/detect_single.py
  ```
  for all images
  ```bash
  python detBalloon/detect_all.py
  ```
</details>







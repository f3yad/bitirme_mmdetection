from ultralytics import YOLO

data_path = "./datasets/spinexr/data.yaml"

model = YOLO("yolov8n.pt") 

model.train(
    data= data_path,
    epochs=100,
    imgsz=800,
    device=0,
    # optimizer="AdamW",


    # lr0=0.01,
    # lrf=0.01,
    # warmup_epochs=3,
    # amp=True,
    save=True,
    save_period=1,
    cache=True,
    resume=False,
    project="bitirme_projesi",
    name="yolov8_classification",
    plots=True,
    #augment=True,
)
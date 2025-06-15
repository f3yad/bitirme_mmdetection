from ultralytics import YOLO

data_path = "./datasets/spinexr/data.yaml"

model = YOLO("yolov8n.pt") 

model.train(
    data= data_path,
    epochs=100,
    batch=16,
    imgsz=224,
    device=0,
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    save=True,
    save_period=10,
    cache=True,
    amp=True,
    resume=False,
    project="bitirme_projesi",
    name="yolov8_classification",
    #augment=True,
    plots=True,
)
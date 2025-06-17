from ultralytics import YOLO
import json

model = YOLO("runs/detect/train2/weights/best.pt")  # Load a pretrained YOLOv8 model
results = model("datasets/spinexr/test_images/", save=False, conf=0.25)  # Predict on a directory of images

predictions = []
for result in results:
    for box in result.boxes:
        predictions.append({
            "image_id": result.path.split("/")[-1],
            "category_id": int(box.cls[0]),
            "bbox": box.xywh[0].tolist(),
            "score": float(box.conf[0])
        })

# Save predictions to a JSON file
with open("f_yolo_predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)
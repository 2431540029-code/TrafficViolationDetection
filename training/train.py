from ultralytics import YOLO

# Load checkpoint đã train đến epoch 13
model = YOLO("runs/detect/helmet_model-2/weights/last.pt")

# Tiếp tục training
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="helmet_model",
    resume=True
)
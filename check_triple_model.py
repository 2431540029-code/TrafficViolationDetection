
from ultralytics import YOLO

# Load model vừa tải từ GitHub
model = YOLO("yolov8s.pt")

print("Danh sach lop mo hinh co the nhan dien:")
for class_id, class_name in model.names.items():
    print(f"ID {class_id}: {class_name}")

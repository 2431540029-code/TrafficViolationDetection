
from ultralytics import YOLO

# Dùng mô hình YOLO COCO có sẵn
model = YOLO("yolo11n.pt")

print("Các lớp YOLO có thể nhận diện:")

for class_id, class_name in model.names.items():
    if class_name in ["person", "motorcycle"]:
        print(f"ID {class_id}: {class_name}")

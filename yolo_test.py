from ultralytics import YOLO
import cv2

# Tải model YOLO
model = YOLO("yolo11n.pt")

# Mở camera
cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc hình ảnh từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc hình ảnh từ camera!")
        break

    # YOLO nhận diện đối tượng
    results = model(frame)

    # Vẽ kết quả nhận diện lên hình ảnh
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLO Traffic Detection", annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng camera
cap.release()
cv2.destroyAllWindows()
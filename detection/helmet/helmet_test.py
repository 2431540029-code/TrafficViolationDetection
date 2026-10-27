from ultralytics import YOLO
import cv2

# Load model helmet đã train
model = YOLO("runs/detect/helmet_model-2/weights/best.pt")

# Mở video
cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video đã kết thúc!")
        break

    # Nhận diện helmet / no_helmet
    results = model(frame, conf=0.5)

    # Vẽ kết quả lên video
    annotated_frame = results[0].plot()

    cv2.imshow("Helmet Detection", annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("✅ Video đã kết thúc!")
        break

    results = model(
        frame,
        classes=[0, 3],   # 0 = person, 3 = motorcycle
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Person + Motorcycle Detection",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
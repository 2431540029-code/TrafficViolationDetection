import cv2

print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc hình ảnh từ camera!")
        break

    cv2.imshow("Traffic Violation Detection", frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
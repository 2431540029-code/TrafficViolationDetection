
from ultralytics import YOLO
import cv2

# ==============================
# CAU HINH
# ==============================
MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/videotest.mp4"

CONFIDENCE = 0.35

# Nguong thu nghiem: tu 3 nguoi tro len
PASSENGER_LIMIT = 2

# Tai mo hinh YOLO
model = YOLO(MODEL_PATH)

# Mo video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Khong mo duoc video: {VIDEO_PATH}")
    raise SystemExit

print("Dang chay V02 - Nhan dien cho qua so nguoi")
print("Nhan Q de thoat.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Phat hien va theo doi nguoi, xe may
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0, 3],
        conf=CONFIDENCE,
        verbose=False
    )

    # Luu thong tin phat hien trong khung hinh hien tai
    people = []
    motorcycles = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            if cls_id == 0:
                people.append({
                    "box": (x1, y1, x2, y2),
                    "id": track_id,
                    "conf": conf
                })

            elif cls_id == 3:
                motorcycles.append({
                    "box": (x1, y1, x2, y2),
                    "id": track_id,
                    "conf": conf
                })

    # Ve nguoi
    for person in people:
        x1, y1, x2, y2 = person["box"]
        label = "Person"

        if person["id"] is not None:
            label += f" ID:{person['id']}"

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (255, 180, 0), 2
        )
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 180, 0), 2
        )

    # Ve xe may va uoc luong so nguoi
    for motorcycle in motorcycles:
        mx1, my1, mx2, my2 = motorcycle["box"]

        # Vung mo rong quanh xe de tim nguoi gan xe
        width = mx2 - mx1
        height = my2 - my1

        rx1 = mx1 - int(width * 0.25)
        rx2 = mx2 + int(width * 0.25)
        ry1 = my1 - int(height * 1.5)
        ry2 = my2 + int(height * 0.15)

        count = 0

        for person in people:
            px1, py1, px2, py2 = person["box"]

            # Tam cua nguoi
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            # Tam nguoi nam trong vung gan xe?
            if rx1 <= pcx <= rx2 and ry1 <= pcy <= ry2:
                count += 1

        # Chi danh dau day la uoc luong ban dau
        if count > PASSENGER_LIMIT:
            color = (0, 0, 255)
            label = f"NGHI VI PHAM: {count} nguoi"
        else:
            color = (0, 220, 0)
            label = f"Xe may: uoc tinh {count} nguoi"

        cv2.rectangle(
            frame, (mx1, my1), (mx2, my2),
            color, 2
        )

        cv2.putText(
            frame, label,
            (mx1, max(25, my1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2
        )

    cv2.imshow("V02 - Passenger Violation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

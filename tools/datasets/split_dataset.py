import os
import shutil
import random

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
TEST_DIR = "dataset/test"

images_dir = os.path.join(TRAIN_DIR, "images")
labels_dir = os.path.join(TRAIN_DIR, "labels")

# Tạo thư mục valid và test
os.makedirs(os.path.join(VALID_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(VALID_DIR, "labels"), exist_ok=True)

os.makedirs(os.path.join(TEST_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "labels"), exist_ok=True)

# Lấy danh sách ảnh
images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Tổng số ảnh:", len(images))

# Trộn ngẫu nhiên
random.seed(42)
random.shuffle(images)

total = len(images)

train_count = int(total * 0.7)
valid_count = int(total * 0.2)

valid_images = images[train_count:train_count + valid_count]
test_images = images[train_count + valid_count:]

print("Giữ lại Train:", train_count)
print("Chuyển sang Valid:", len(valid_images))
print("Chuyển sang Test:", len(test_images))


def move_files(image_list, destination):
    for image_name in image_list:

        # Đường dẫn ảnh
        image_source = os.path.join(images_dir, image_name)

        # Tên file label tương ứng
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_source = os.path.join(labels_dir, label_name)

        # Chuyển ảnh
        shutil.move(
            image_source,
            os.path.join(destination, "images", image_name)
        )

        # Chuyển label
        if os.path.exists(label_source):
            shutil.move(
                label_source,
                os.path.join(destination, "labels", label_name)
            )
        else:
            print("CẢNH BÁO: Không tìm thấy label:", label_name)


# Chuyển 20% sang validation
move_files(valid_images, VALID_DIR)

# Chuyển 10% sang test
move_files(test_images, TEST_DIR)

print("\n================================")
print("CHIA DATASET HOÀN TẤT!")
print("================================")
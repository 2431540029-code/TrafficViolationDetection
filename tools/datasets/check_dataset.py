import os

folders = [
    "dataset/train/images",
    "dataset/train/labels",
    "dataset/valid/images",
    "dataset/valid/labels",
    "dataset/test/images",
    "dataset/test/labels"
]

print("========== KIỂM TRA DATASET ==========")

for folder in folders:
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"{folder}: {len(files)} file")
    else:
        print(f"❌ KHÔNG TỒN TẠI: {folder}")

print("======================================")
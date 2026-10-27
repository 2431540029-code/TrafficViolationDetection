from huggingface_hub import hf_hub_download

print("Đang tải model nhận diện biển số...")

model_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

print("================================")
print("TẢI MODEL THÀNH CÔNG!")
print("Model nằm tại:")
print(model_path)
print("================================")
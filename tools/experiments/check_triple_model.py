
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]

model_path = (
    BASE_DIR
    / "runs"
    / "detect"
    / "v02_trippling"
    / "weights"
    / "best.pt"
)

model = YOLO(str(model_path))

print("Ten cac lop trong model:")
print(model.names)

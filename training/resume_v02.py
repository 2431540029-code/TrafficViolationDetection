
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

LAST_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "v02_trippling"
    / "weights"
    / "last.pt"
)

if not LAST_MODEL.exists():
    raise FileNotFoundError(
        f"Khong tim thay checkpoint: {LAST_MODEL}"
    )

print("Dang tiep tuc train tu checkpoint:")
print(LAST_MODEL)

model = YOLO(str(LAST_MODEL))
model.train(resume=True)

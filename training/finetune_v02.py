
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_DIR / "dataset_v02_yolo" / "data.yaml"
BEST_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "v02_trippling"
    / "weights"
    / "best.pt"
)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Khong tim thay dataset: {DATA_YAML}")

if not BEST_MODEL.exists():
    raise FileNotFoundError(f"Khong tim thay model: {BEST_MODEL}")

model = YOLO(str(BEST_MODEL))

model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project=str(PROJECT_DIR / "runs" / "detect"),
    name="v02_finetune",
    patience=5,
    save=True,
    plots=True
)

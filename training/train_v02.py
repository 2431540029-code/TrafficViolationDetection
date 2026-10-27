
from ultralytics import YOLO
from pathlib import Path

# Thu muc project hien tai
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Duong dan dataset da chuyen sang YOLO
DATA_YAML = PROJECT_DIR / "dataset_v02_yolo" / "data.yaml"

# Kiem tra du lieu va trong so
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Khong tim thay: {DATA_YAML}")

WEIGHTS = PROJECT_DIR / "yolo11n.pt"

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"Khong tim thay {WEIGHTS}. "
        "Hay kiem tra file yolo11n.pt trong thu muc project."
    )

# Load YOLO11 nano
model = YOLO(str(WEIGHTS))

# Bat dau huan luyen V02
model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project=str(PROJECT_DIR / "runs" / "detect"),
    name="v02_trippling",
    patience=10,
    save=True,
    plots=True
)

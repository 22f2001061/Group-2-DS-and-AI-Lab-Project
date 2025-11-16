from ultralytics import YOLO
from config.settings import MODEL_PATH, DEVICE

print(f"Loading YOLO model from {MODEL_PATH}...")
yolo = YOLO(MODEL_PATH)

try:
    yolo.to(DEVICE)
except Exception:
    pass

PERSON_CLASS_ID = 0
try:
    if hasattr(yolo, "names") and isinstance(yolo.names, dict):
        for k, v in yolo.names.items():
            if str(v).lower() == "person":
                PERSON_CLASS_ID = int(k)
                break
except Exception:
    PERSON_CLASS_ID = 0

print("Detected person class id:", PERSON_CLASS_ID)

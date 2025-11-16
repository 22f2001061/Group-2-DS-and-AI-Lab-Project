import os

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
TTS_CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

STATIC_DIR = "static"

DEFAULT_PERSON_HEIGHT_M = float(os.environ.get("DEFAULT_PERSON_HEIGHT_M", 1.7))
FOCAL_LENGTH_PX = float(os.environ.get("FOCAL_LENGTH_PX", 1000.0))
ALERT_DISTANCE_PERSON_M = float(os.environ.get("ALERT_DISTANCE_PERSON_M", 25))
ALERT_DISTANCE_OBJECT_M = float(os.environ.get("ALERT_DISTANCE_OBJECT_M", 25))

ALERT_CLASS_COOLDOWN_SEC = float(os.environ.get("ALERT_CLASS_COOLDOWN_SEC", 3.0))
ALERT_REPEAT_DELAY_SEC = float(os.environ.get("ALERT_REPEAT_DELAY_SEC", 5.0))
ALERT_GLOBAL_COOLDOWN_SEC = float(os.environ.get("ALERT_GLOBAL_COOLDOWN_SEC", 0.3))

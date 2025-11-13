import cv2
import torch
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt', device=None, img_size=640, conf=0.4):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.img_size = img_size
        self.conf = conf

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.img_size / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return self.model.track(frame, conf=self.conf, persist=True, verbose=False)

import time
import json
import cv2
import numpy as np
from aiortc import MediaStreamTrack
from inference.yolo_loader import yolo, PERSON_CLASS_ID
from inference.alert_manager import AlertManager
from utils.distance import estimate_distance_px
from utils.helpers import extract_boxes_cls_ids_from_result
from config.settings import (
    DEFAULT_PERSON_HEIGHT_M,
    ALERT_DISTANCE_PERSON_M,
    ALERT_DISTANCE_OBJECT_M,
)

class InferenceVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, data_channel_holder):
        super().__init__()
        self.track = track
        self.data_channel_holder = data_channel_holder
        self.alert_mgr = AlertManager()
        self.frame_count = 0
        self.pending_alerts = []

    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1

        if self.frame_count % 10 == 0:
            self._try_send_pending_alerts()

        if self.frame_count % 3 != 0:
            return frame

        img = frame.to_ndarray(format="bgr24")
        results = yolo.track(img, conf=0.35, persist=True, verbose=False)

        if not results:
            return frame

        xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(results[0])
        if xyxy is None:
            return frame

        h, w = img.shape[:2]
        potential = []
        now = time.time()

        for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
            x1, y1, x2, y2 = map(int, box[:4])
            box_h_px = max(1, y2 - y1)
            object_real_h = DEFAULT_PERSON_HEIGHT_M
            dist = estimate_distance_px(box_h_px, object_real_h)

            cx = (x1 + x2) // 2
            rel = (cx - (w / 2)) / (w / 2)
            direction = "right" if rel > 0.25 else ("left" if rel < -0.25 else "ahead")

            class_name = yolo.names.get(int(cls_id), f"Class{int(cls_id)}")

            is_person = cls_id == PERSON_CLASS_ID
            is_close = (is_person and dist < ALERT_DISTANCE_PERSON_M) or \
                       ((not is_person) and dist < ALERT_DISTANCE_OBJECT_M)

            if is_close and self.alert_mgr.can_alert(track_id, cls_id, now):
                potential.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "distance": dist,
                    "direction": direction
                })

        if not potential:
            self._try_send_pending_alerts()
            return frame

        potential.sort(key=lambda x: (0 if x["class_id"] == PERSON_CLASS_ID else 1, x["distance"]))
        chosen = potential[0]

        text = f"{chosen['class_name']}, {chosen['distance']:.1f} meters, {chosen['direction']}"
        timestamp_ms = int(time.time() * 1000)
        self.alert_mgr.register_alert(chosen["track_id"], chosen["class_id"], timestamp_ms)

        payload = {
            "type": "alert",
            "text": text,
            "timestamp_ms": timestamp_ms,
            "distance": chosen["distance"],
            "direction": chosen["direction"],
            "class_name": chosen["class_name"]
        }

        self._send_alert(payload)
        return frame

    def _send_alert(self, payload):
        data_channel = self.data_channel_holder.get("channel")
        try:
            if data_channel and data_channel.readyState == "open":
                data_channel.send(json.dumps(payload))
                self._try_send_pending_alerts()
            else:
                self.pending_alerts.append(payload)
        except Exception:
            if payload not in self.pending_alerts:
                self.pending_alerts.append(payload)

    def _try_send_pending_alerts(self):
        data_channel = self.data_channel_holder.get("channel")
        if not data_channel or data_channel.readyState != "open":
            return

        failed = []
        while self.pending_alerts:
            alert = self.pending_alerts.pop(0)
            try:
                data_channel.send(json.dumps(alert))
            except Exception:
                failed.append(alert)
                break

        self.pending_alerts = failed + self.pending_alerts

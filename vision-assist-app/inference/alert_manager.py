import time
from config.settings import (
    ALERT_CLASS_COOLDOWN_SEC,
    ALERT_REPEAT_DELAY_SEC,
    ALERT_GLOBAL_COOLDOWN_SEC,
)

class AlertManager:
    def __init__(self, per_class_cd=ALERT_CLASS_COOLDOWN_SEC,
                 repeat_delay=ALERT_REPEAT_DELAY_SEC,
                 global_cd=ALERT_GLOBAL_COOLDOWN_SEC):
        self.per_class_cd = per_class_cd
        self.repeat_delay = repeat_delay
        self.global_cd = global_cd
        self.last_class_time = {}
        self.last_track_time = {}
        self.last_global_time = -1e9

    def can_alert(self, track_id: int, class_id: int, now_s: float) -> bool:
        if now_s - self.last_global_time < self.global_cd:
            return False
        if now_s - self.last_class_time.get(class_id, -1e9) < self.per_class_cd:
            return False
        if now_s - self.last_track_time.get(track_id, -1e9) < self.repeat_delay:
            return False
        return True

    def register_alert(self, track_id: int, class_id: int, timestamp_ms: int):
        now_s = timestamp_ms / 1000.0
        self.last_global_time = now_s
        self.last_class_time[class_id] = now_s
        self.last_track_time[track_id] = now_s

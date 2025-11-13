import time

class AlertManager:
    def __init__(self, per_class_cd=8.0, repeat_delay=15.0, global_cd=0.0):
        self.per_class_cd = per_class_cd
        self.repeat_delay = repeat_delay
        self.global_cd = global_cd
        self.last_class_time = {}
        self.last_track_time = {}
        self.last_global_time = -1e9
        self.alert_log = []

    def can_alert(self, track_id, class_id):
        now = time.time()
        if now - self.last_global_time < self.global_cd: return False
        if now - self.last_class_time.get(class_id, -1e9) < self.per_class_cd: return False
        if now - self.last_track_time.get(track_id, -1e9) < self.repeat_delay: return False
        return True

    def register_alert(self, track_id, class_id, text, timestamp_ms):
        now = time.time()
        self.last_global_time = now
        self.last_class_time[class_id] = now
        self.last_track_time[track_id] = now
        self.alert_log.append((timestamp_ms, text))

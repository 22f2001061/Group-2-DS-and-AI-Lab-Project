from collections import deque

class ObjectTracker:
    def __init__(self, maxlen=15, movement_thresh_px=5, direction_ratio=0.3):
        self.hist = {}
        self.maxlen = maxlen
        self.movement_thresh_px = movement_thresh_px
        self.direction_ratio = direction_ratio

    def update(self, track_id, cx, cy):
        if track_id not in self.hist:
            self.hist[track_id] = deque(maxlen=self.maxlen)
        self.hist[track_id].append((cx, cy))

    def get_motion_direction(self, track_id, frame_width):
        if track_id not in self.hist or len(self.hist[track_id]) < 2:
            return "Ahead", "Static"

        history = self.hist[track_id]
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        motion = "Moving" if abs(dx) > self.movement_thresh_px or abs(dy) > self.movement_thresh_px else "Static"

        relative = (history[-1][0] - (frame_width / 2)) / (frame_width / 2)
        direction = "Right" if relative > self.direction_ratio else ("Left" if relative < -self.direction_ratio else "Ahead")
        return direction, motion

    def remove_missing(self, active_ids):
        for tid in list(self.hist.keys()):
            if tid not in active_ids:
                del self.hist[tid]

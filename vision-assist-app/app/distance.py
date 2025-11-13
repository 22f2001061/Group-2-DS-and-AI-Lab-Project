import math

class DistanceEstimator:
    def __init__(self, focal_length_px=1000.0, default_height_m=1.6, class_heights=None):
        self.focal_px = focal_length_px
        self.default_h = default_height_m
        self.class_heights = class_heights or {}

    def estimate(self, box_height_px, class_id=None):
        if box_height_px <= 0:
            return float('inf')
        real_height = self.class_heights.get(class_id, self.default_h)
        return (real_height * self.focal_px) / box_height_px

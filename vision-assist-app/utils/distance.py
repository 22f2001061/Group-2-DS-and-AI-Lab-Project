import math
from config.settings import FOCAL_LENGTH_PX

def estimate_distance_px(box_h_px: float, object_real_h_m: float, focal_px: float = FOCAL_LENGTH_PX) -> float:
    if box_h_px <= 0 or math.isinf(box_h_px):
        return float("inf")
    return (object_real_h_m * focal_px) / float(box_h_px)

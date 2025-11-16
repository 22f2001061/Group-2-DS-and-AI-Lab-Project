import hashlib
import numpy as np

def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_boxes_cls_ids_from_result(result):
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return None, None, None

    xyxy_tensor = getattr(boxes_obj, "xyxy", None)
    if xyxy_tensor is None:
        return None, None, None

    try:
        xyxy = xyxy_tensor.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array(xyxy_tensor)
        except Exception:
            xyxy = None

    if xyxy is None or xyxy.size == 0:
        return None, None, None

    cls_tensor = getattr(boxes_obj, "cls", None)
    try:
        cls_ids = cls_tensor.cpu().numpy().astype(int)
    except Exception:
        try:
            cls_ids = np.array(cls_tensor).astype(int)
        except Exception:
            cls_ids = np.zeros((xyxy.shape[0],), dtype=int)

    id_tensor = getattr(boxes_obj, "id", None)
    try:
        if id_tensor is None:
            track_ids = np.arange(0, xyxy.shape[0], dtype=int)
        else:
            track_ids = id_tensor.cpu().numpy().astype(int)
    except Exception:
        try:
            track_ids = np.array(id_tensor).astype(int)
        except Exception:
            track_ids = np.arange(0, xyxy.shape[0], dtype=int)

    return xyxy, cls_ids, track_ids

# server_fastapi.py
import base64
import io
import os
import time
import json
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# gTTS used as default TTS in the demo (network). Swap with local TTS for production/low-latency.
from gtts import gTTS

# Ultralytics YOLO import
from ultralytics import YOLO

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cuda")  # let YOLO.to handle if cuda not available
TTS_CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Distance / alert thresholds
DEFAULT_PERSON_HEIGHT_M = float(os.environ.get("DEFAULT_PERSON_HEIGHT_M", 1.7))
FOCAL_LENGTH_PX = float(os.environ.get("FOCAL_LENGTH_PX", 1000.0))
ALERT_DISTANCE_PERSON_M = float(os.environ.get("ALERT_DISTANCE_PERSON_M", 2.5))
ALERT_DISTANCE_OBJECT_M = float(os.environ.get("ALERT_DISTANCE_OBJECT_M", 5.0))

# Cooldowns
ALERT_CLASS_COOLDOWN_SEC = float(os.environ.get("ALERT_CLASS_COOLDOWN_SEC", 6.0))
ALERT_REPEAT_DELAY_SEC = float(os.environ.get("ALERT_REPEAT_DELAY_SEC", 10.0))
ALERT_GLOBAL_COOLDOWN_SEC = float(os.environ.get("ALERT_GLOBAL_COOLDOWN_SEC", 0.5))

# Thread pool for blocking work (TTS)
THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# ----------------------
# Helper functions
# ----------------------
def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def text_to_mp3_bytes_with_cache(text: str) -> bytes:
    """
    Blocking TTS helper using gTTS and a file cache.
    Replace this with a local TTS model for lower latency.
    """
    key = _text_hash(text)
    cache_path = os.path.join(TTS_CACHE_DIR, f"{key}.mp3")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    # synthesize (blocking)
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    mp3 = buf.getvalue()
    with open(cache_path, "wb") as f:
        f.write(mp3)
    return mp3

def estimate_distance_px(box_h_px: float, object_real_h_m: float, focal_px: float = FOCAL_LENGTH_PX) -> float:
    if box_h_px <= 0 or math.isinf(box_h_px):
        return float("inf")
    return (object_real_h_m * focal_px) / box_h_px

# ----------------------
# Alert manager class
# ----------------------
class AlertManager:
    def __init__(self, per_class_cd=ALERT_CLASS_COOLDOWN_SEC, repeat_delay=ALERT_REPEAT_DELAY_SEC, global_cd=ALERT_GLOBAL_COOLDOWN_SEC):
        self.per_class_cd = per_class_cd
        self.repeat_delay = repeat_delay
        self.global_cd = global_cd
        self.last_class_time: Dict[int, float] = {}
        self.last_track_time: Dict[int, float] = {}
        self.last_global_time: float = -1e9

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

# ----------------------
# Load YOLO model (robust)
# ----------------------
print(f"Loading YOLO model {MODEL_PATH} ...")
yolo = YOLO(MODEL_PATH)
try:
    yolo.to(DEVICE)
except Exception:
    # ignore if device transfer not supported; YOLO may handle internally
    pass

# find person class id if available
PERSON_CLASS_ID = 0
try:
    if hasattr(yolo, "names") and isinstance(yolo.names, dict):
        for k, v in yolo.names.items():
            if str(v).lower() == "person":
                PERSON_CLASS_ID = int(k)
                break
except Exception:
    PERSON_CLASS_ID = 0
print(f"Person class id: {PERSON_CLASS_ID}")

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return HTMLResponse(open("static/client.html", "r", encoding="utf-8").read())

# ----------------------
# Utility: safe extraction of boxes, cls, ids from result
# Returns (xyxy_np, cls_np, id_np) or (None, None, None) if no boxes
# ----------------------
def extract_boxes_cls_ids_from_result(result) -> (Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]):
    """
    Accepts a single ultralytics result object and extracts numpy arrays for boxes, classes and track ids.
    Returns (xyxy, cls, ids) or (None, None, None) if nothing to extract.
    """
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return None, None, None

    # xyxy
    xyxy_tensor = getattr(boxes_obj, "xyxy", None)
    if xyxy_tensor is None:
        return None, None, None
    # try to move to cpu and convert to numpy safely
    try:
        xyxy = xyxy_tensor.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array(xyxy_tensor)
        except Exception:
            return None, None, None
    if xyxy is None or xyxy.size == 0:
        return None, None, None

    # class ids
    cls_tensor = getattr(boxes_obj, "cls", None)
    try:
        cls_ids = cls_tensor.cpu().numpy().astype(int) if cls_tensor is not None else np.zeros((xyxy.shape[0],), dtype=int)
    except Exception:
        try:
            cls_ids = np.array(cls_tensor).astype(int) if cls_tensor is not None else np.zeros((xyxy.shape[0],), dtype=int)
        except Exception:
            cls_ids = np.zeros((xyxy.shape[0],), dtype=int)

    # track ids (may be None if not tracking)
    id_tensor = getattr(boxes_obj, "id", None)
    try:
        if id_tensor is None:
            # create synthetic track ids (unique per frame) to avoid None issues
            track_ids = np.arange(0, xyxy.shape[0], dtype=int)
        else:
            track_ids = id_tensor.cpu().numpy().astype(int)
    except Exception:
        try:
            track_ids = np.array(id_tensor).astype(int) if id_tensor is not None else np.arange(0, xyxy.shape[0], dtype=int)
        except Exception:
            track_ids = np.arange(0, xyxy.shape[0], dtype=int)

    return xyxy, cls_ids, track_ids

# ----------------------
# WebSocket endpoint for real-time frames (camera)
# ----------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected (websocket camera).")
    alert_mgr = AlertManager()
    try:
        while True:
            msg = await ws.receive_text()
            # client can send either JSON {"type":"frame","b64":...} or raw dataURL
            try:
                payload = json.loads(msg)
                if payload.get("type") == "frame" and "b64" in payload:
                    data_url = payload["b64"]
                else:
                    data_url = msg
            except Exception:
                data_url = msg

            # decode base64 image
            if "," in data_url:
                b64 = data_url.split(",", 1)[1]
            else:
                b64 = data_url
            try:
                frame_bytes = base64.b64decode(b64)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await ws.send_json({"type": "error", "message": "invalid_frame"})
                continue

            if frame is None:
                await ws.send_json({"type": "error", "message": "invalid_frame_decode"})
                continue

            # === run YOLO tracker/inference (robust) ===
            try:
                results = yolo.track(frame, conf=0.35, persist=True, verbose=False)
            except Exception as e:
                print("YOLO error:", e)
                await ws.send_json({"type": "error", "message": "detector_error"})
                continue

            if not results or len(results) == 0:
                # nothing detected this frame
                continue

            res0 = results[0]
            xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
            if xyxy is None:
                # no boxes
                continue

            frame_h, frame_w = frame.shape[:2]
            timestamp_ms = int(time.time() * 1000)

            potential_alerts = []
            for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
                # ensure ints
                try:
                    x1, y1, x2, y2 = map(int, box[:4])
                except Exception:
                    continue
                box_h_px = max(1, (y2 - y1))
                # choose expected height (simple)
                object_real_h = DEFAULT_PERSON_HEIGHT_M if int(cls_id) == PERSON_CLASS_ID else DEFAULT_PERSON_HEIGHT_M
                est_dist = estimate_distance_px(box_h_px, object_real_h)

                # direction: based on center relative to frame center (tunable)
                cx = (x1 + x2) // 2
                rel = (cx - (frame_w / 2)) / (frame_w / 2)
                if rel > 0.25:
                    direction = "Right"
                elif rel < -0.25:
                    direction = "Left"
                else:
                    direction = "Ahead"

                class_name = yolo.names.get(int(cls_id), f"Class{cls_id}")
                is_person = (int(cls_id) == PERSON_CLASS_ID)
                is_close_person = is_person and (est_dist < ALERT_DISTANCE_PERSON_M)
                is_close_object = (not is_person) and (est_dist < ALERT_DISTANCE_OBJECT_M)

                if (is_close_person or is_close_object) and alert_mgr.can_alert(int(track_id), int(cls_id), time.time()):
                    potential_alerts.append({
                        "track_id": int(track_id),
                        "class_id": int(cls_id),
                        "class_name": str(class_name),
                        "distance": float(est_dist),
                        "direction": direction
                    })

            if not potential_alerts:
                continue

            # prioritize persons first, then by distance ascending
            potential_alerts.sort(key=lambda x: (0 if x['class_id'] == PERSON_CLASS_ID else 1, x['distance']))
            chosen = potential_alerts[0]
            dist_str = f"{chosen['distance']:.1f}"
            alert_text = f"Caution: {chosen['class_name']}, about {dist_str} meters, {chosen['direction']}."

            # register and produce TTS on threadpool
            timestamp_ms = int(time.time() * 1000)
            alert_mgr.register_alert(chosen['track_id'], chosen['class_id'], timestamp_ms)

            loop = asyncio.get_event_loop()
            mp3_bytes = await loop.run_in_executor(THREAD_POOL, text_to_mp3_bytes_with_cache, alert_text)
            b64 = base64.b64encode(mp3_bytes).decode("utf-8")

            await ws.send_json({
                "type": "audio",
                "text": alert_text,
                "audio_b64": b64,
                "timestamp_ms": timestamp_ms
            })

    except Exception as e:
        print("WebSocket ended (camera).", e)
    finally:
        print("WebSocket client disconnected (camera).")

# ----------------------
# Upload-video endpoint -> SSE streaming of alerts
# ----------------------
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts multipart upload of a video file and returns an SSE stream of alert events.
    Each event data is JSON: { text, audio_b64, timestamp_ms }.
    """

    # save uploaded file temporarily
    contents = await file.read()
    tmp_path = f"temp_upload_{int(time.time()*1000)}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    async def event_generator():
        """
        Process the video and yield SSE events as JSON lines:
        event: alert\n
        data: <json>\n\n
        """

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield f"event: error\ndata: {json.dumps({'message':'could_not_open_video'})}\n\n"
            os.remove(tmp_path)
            return

        alert_mgr = AlertManager()
        frame_idx = 0

        # Use persistent YOLO tracking across frames (persist=True)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # run YOLO track (wrap in try)
                try:
                    results = yolo.track(frame, conf=0.35, persist=True, verbose=False)
                except Exception as ex:
                    # send error event and continue
                    yield f"event: error\ndata: {json.dumps({'message':'detector_error','detail':str(ex)})}\n\n"
                    continue

                if not results or len(results) == 0:
                    continue
                res0 = results[0]
                xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
                if xyxy is None:
                    continue

                frame_h, frame_w = frame.shape[:2]
                potential_alerts = []
                timestamp_ms = int(time.time() * 1000)

                for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
                    try:
                        x1, y1, x2, y2 = map(int, box[:4])
                    except Exception:
                        continue
                    box_h_px = max(1, (y2 - y1))
                    object_real_h = DEFAULT_PERSON_HEIGHT_M if int(cls_id) == PERSON_CLASS_ID else DEFAULT_PERSON_HEIGHT_M
                    est_dist = estimate_distance_px(box_h_px, object_real_h)

                    cx = (x1 + x2) // 2
                    rel = (cx - (frame_w / 2)) / (frame_w / 2)
                    direction = "Right" if rel > 0.25 else ("Left" if rel < -0.25 else "Ahead")

                    class_name = yolo.names.get(int(cls_id), f"Class{cls_id}")
                    is_person = (int(cls_id) == PERSON_CLASS_ID)
                    is_close_person = is_person and (est_dist < ALERT_DISTANCE_PERSON_M)
                    is_close_object = (not is_person) and (est_dist < ALERT_DISTANCE_OBJECT_M)

                    if (is_close_person or is_close_object) and alert_mgr.can_alert(int(track_id), int(cls_id), time.time()):
                        potential_alerts.append({
                            "track_id": int(track_id),
                            "class_id": int(cls_id),
                            "class_name": str(class_name),
                            "distance": float(est_dist),
                            "direction": direction
                        })

                if not potential_alerts:
                    continue

                potential_alerts.sort(key=lambda x: (0 if x['class_id'] == PERSON_CLASS_ID else 1, x['distance']))
                chosen = potential_alerts[0]
                dist_str = f"{chosen['distance']:.1f}"
                alert_text = f"Caution: {chosen['class_name']}, about {dist_str} meters, {chosen['direction']}."

                timestamp_ms = int(time.time() * 1000)
                alert_mgr.register_alert(chosen['track_id'], chosen['class_id'], timestamp_ms)

                # synthesize audio in threadpool
                loop = asyncio.get_event_loop()
                try:
                    mp3_bytes = await loop.run_in_executor(THREAD_POOL, text_to_mp3_bytes_with_cache, alert_text)
                except Exception as e:
                    # fallback to sending only text
                    yield f"event: alert\ndata: {json.dumps({'text': alert_text, 'audio_b64': None, 'timestamp_ms': timestamp_ms})}\n\n"
                    continue

                b64 = base64.b64encode(mp3_bytes).decode("utf-8")
                payload = {"text": alert_text, "audio_b64": b64, "timestamp_ms": timestamp_ms}
                # SSE event
                yield f"event: alert\ndata: {json.dumps(payload)}\n\n"

            # finished processing
            yield f"event: done\ndata: {json.dumps({'message':'finished'})}\n\n"

        finally:
            cap.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ----------------------
# End of file
# ----------------------

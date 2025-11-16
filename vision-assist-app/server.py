import os
import io
import time
import json
import math
import base64
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

# aiortc
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

# YOLO
from ultralytics import YOLO

# TTS backends
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

# config
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
TTS_CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

DEFAULT_PERSON_HEIGHT_M = float(os.environ.get("DEFAULT_PERSON_HEIGHT_M", 1.7))
FOCAL_LENGTH_PX = float(os.environ.get("FOCAL_LENGTH_PX", 1000.0))
ALERT_DISTANCE_PERSON_M = float(os.environ.get("ALERT_DISTANCE_PERSON_M", 25))
ALERT_DISTANCE_OBJECT_M = float(os.environ.get("ALERT_DISTANCE_OBJECT_M", 25))

# Reduced cooldowns for more responsive feedback
ALERT_CLASS_COOLDOWN_SEC = float(os.environ.get("ALERT_CLASS_COOLDOWN_SEC", 3.0))
ALERT_REPEAT_DELAY_SEC = float(os.environ.get("ALERT_REPEAT_DELAY_SEC", 5.0))
ALERT_GLOBAL_COOLDOWN_SEC = float(os.environ.get("ALERT_GLOBAL_COOLDOWN_SEC", 0.3))

THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# helpers
def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def text_to_audio_bytes_with_cache(text: str) -> Tuple[bytes, str]:
    """Generate TTS audio with caching. Always returns WAV for consistency."""
    key = _text_hash(text)
    cache_wav = os.path.join(TTS_CACHE_DIR, f"{key}.wav")

    # Check cache first
    if os.path.exists(cache_wav):
        with open(cache_wav, "rb") as f:
            return f.read(), "audio/wav"

    # Try pyttsx3 first (faster, offline)
    if HAS_PYTTSX3:
        try:
            engine = pyttsx3.init()
            # Faster speech rate for quicker alerts
            engine.setProperty('rate', 180)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            engine.save_to_file(text, tmp)
            engine.runAndWait()
            with open(tmp, "rb") as f:
                data = f.read()
            with open(cache_wav, "wb") as f:
                f.write(data)
            try:
                os.remove(tmp)
            except Exception:
                pass
            return data, "audio/wav"
        except Exception as e:
            print(f"pyttsx3 error: {e}")

    # Fallback to gTTS
    if HAS_GTTS:
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            # Convert to WAV for consistency
            fd, tmp_mp3 = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tts.save(tmp_mp3)
            
            # Convert MP3 to WAV using ffmpeg if available, otherwise just use MP3
            try:
                import subprocess
                fd2, tmp_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd2)
                subprocess.run(['ffmpeg', '-i', tmp_mp3, '-ar', '16000', '-ac', '1', tmp_wav], 
                             check=True, capture_output=True)
                with open(tmp_wav, "rb") as f:
                    data = f.read()
                os.remove(tmp_wav)
                os.remove(tmp_mp3)
                with open(cache_wav, "wb") as f:
                    f.write(data)
                return data, "audio/wav"
            except Exception:
                # Just use MP3 if conversion fails
                with open(tmp_mp3, "rb") as f:
                    data = f.read()
                os.remove(tmp_mp3)
                return data, "audio/mpeg"
        except Exception as e:
            print(f"gTTS error: {e}")

    return b"", "application/octet-stream"

def estimate_distance_px(box_h_px: float, object_real_h_m: float, focal_px: float = FOCAL_LENGTH_PX) -> float:
    if box_h_px <= 0 or math.isinf(box_h_px):
        return float("inf")
    return (object_real_h_m * focal_px) / float(box_h_px)

class AlertManager:
    def __init__(self, per_class_cd=ALERT_CLASS_COOLDOWN_SEC, repeat_delay=ALERT_REPEAT_DELAY_SEC, global_cd=ALERT_GLOBAL_COOLDOWN_SEC):
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
        cls_ids = cls_tensor.cpu().numpy().astype(int) if cls_tensor is not None else np.zeros((xyxy.shape[0],), dtype=int)
    except Exception:
        try:
            cls_ids = np.array(cls_tensor).astype(int) if cls_tensor is not None else np.zeros((xyxy.shape[0],), dtype=int)
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
            track_ids = np.array(id_tensor).astype(int) if id_tensor is not None else np.arange(0, xyxy.shape[0], dtype=int)
        except Exception:
            track_ids = np.arange(0, xyxy.shape[0], dtype=int)
    return xyxy, cls_ids, track_ids

print(f"Loading YOLO model from {MODEL_PATH} ...")
yolo = YOLO(MODEL_PATH)
try:
    yolo.to(DEVICE)
except Exception:
    pass

PERSON_CLASS_ID = 0
try:
    if hasattr(yolo, "names") and isinstance(yolo.names, dict):
        for k, v in yolo.names.items():
            if str(v).lower() == "person":
                PERSON_CLASS_ID = int(k)
                break
except Exception:
    PERSON_CLASS_ID = 0
print("Person class id:", PERSON_CLASS_ID)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

class Offer(BaseModel):
    sdp: str
    type: str

pcs = set()

class InferenceVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track, data_channel_holder):
        super().__init__()
        self.track = track
        self.data_channel_holder = data_channel_holder  # Store reference to holder dict
        self.alert_mgr = AlertManager()
        self.frame_count = 0
        self.pending_alerts = []  # Initialize pending alerts queue
        
    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1
        
        # Try to send pending alerts every frame
        if self.frame_count % 10 == 0:  # Every 10 frames
            self._try_send_pending_alerts()
        
        # Process every 3rd frame for performance
        if self.frame_count % 3 != 0:
            return frame
        
        if self.frame_count % 30 == 0:  # Log every 30 frames (~1 second)
            print(f"Processing frame {self.frame_count}... (Pending alerts: {len(self.pending_alerts)})")
            
        img = frame.to_ndarray(format="bgr24")
        try:
            results = yolo.track(img, conf=0.35, persist=True, verbose=False)
        except Exception as e:
            print(f"YOLO error: {e}")
            return frame
            
        if not results or len(results) == 0:
            return frame
            
        res0 = results[0]
        xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
        if xyxy is None or len(xyxy) == 0:
            return frame
        
        if self.frame_count % 30 == 0:
            print(f"  Detected {len(xyxy)} objects")
            
        h, w = img.shape[:2]
        potential = []
        now = time.time()
        
        for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
            try:
                x1, y1, x2, y2 = map(int, box[:4])
            except Exception:
                continue
                
            box_h_px = max(1, y2 - y1)
            object_real_h = DEFAULT_PERSON_HEIGHT_M if int(cls_id) == PERSON_CLASS_ID else DEFAULT_PERSON_HEIGHT_M
            dist = estimate_distance_px(box_h_px, object_real_h)
            
            cx = (x1 + x2) // 2
            rel = (cx - (w / 2)) / (w / 2)
            direction = "right" if rel > 0.25 else ("left" if rel < -0.25 else "ahead")
            
            class_name = yolo.names.get(int(cls_id), f"Class{int(cls_id)}")
            is_person = (int(cls_id) == PERSON_CLASS_ID)
            is_close_person = is_person and (dist < ALERT_DISTANCE_PERSON_M)
            is_close_object = (not is_person) and (dist < ALERT_DISTANCE_OBJECT_M)
            
            if (is_close_person or is_close_object) and self.alert_mgr.can_alert(int(track_id), int(cls_id), now):
                potential.append({
                    "track_id": int(track_id), 
                    "class_id": int(cls_id), 
                    "class_name": str(class_name), 
                    "distance": float(dist), 
                    "direction": direction
                })
                
        if not potential:
            # Try to send any pending alerts even if no new detection
            self._try_send_pending_alerts()
            return frame
            
        # Prioritize person detections and closer objects
        potential.sort(key=lambda x: (0 if x["class_id"] == PERSON_CLASS_ID else 1, x["distance"]))
        chosen = potential[0]
        
        # Shorter, clearer message for TTS
        text = f"{chosen['class_name']}, {chosen['distance']:.1f} meters, {chosen['direction']}"
        timestamp_ms = int(time.time() * 1000)
        self.alert_mgr.register_alert(chosen['track_id'], chosen['class_id'], timestamp_ms)
        
        # Send only text to frontend (TTS will be done client-side)
        payload = {
            "type": "alert", 
            "text": text, 
            "timestamp_ms": timestamp_ms,
            "distance": chosen['distance'],
            "direction": chosen['direction'],
            "class_name": chosen['class_name']
        }
        
        print(f"[ALERT] {text}")
        self._send_alert(payload)
        return frame
    
    def _send_alert(self, payload):
        """Send alert via datachannel with queueing if not ready"""
        data_channel = self.data_channel_holder.get("channel")
        
        try:
            if data_channel and data_channel.readyState == "open":
                data_channel.send(json.dumps(payload))
                print(f"  ✓ Alert sent via DataChannel")
                # Also try to send any pending alerts
                self._try_send_pending_alerts()
            else:
                state = data_channel.readyState if data_channel else 'None'
                print(f"  ✗ DataChannel not ready (state: {state})")
                self.pending_alerts.append(payload)
                print(f"  → Queued alert (total queued: {len(self.pending_alerts)})")
        except Exception as e:
            print(f"  ✗ DataChannel send error: {e}")
            # Queue the alert for retry
            if payload not in self.pending_alerts:
                self.pending_alerts.append(payload)
    
    def _try_send_pending_alerts(self):
        """Try to send any pending alerts if datachannel is ready"""
        if not self.pending_alerts:
            return
            
        data_channel = self.data_channel_holder.get("channel")
        if not data_channel or data_channel.readyState != "open":
            return
            
        sent_count = 0
        failed_alerts = []
        
        while self.pending_alerts:
            try:
                pending = self.pending_alerts.pop(0)
                data_channel.send(json.dumps(pending))
                sent_count += 1
            except Exception as e:
                print(f"  ✗ Failed to send pending alert: {e}")
                failed_alerts.append(pending)
                break
        
        # Re-add failed alerts
        self.pending_alerts = failed_alerts + self.pending_alerts
        
        if sent_count > 0:
            print(f"  ✓ Sent {sent_count} pending alert(s)")

@app.post("/offer")
async def offer(sdp: Offer):
    print("=" * 50)
    print("Received WebRTC offer")
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    data_channel_holder = {"channel": None}
    
    @pc.on("datachannel")
    def on_datachannel(channel):
        data_channel_holder["channel"] = channel
        print(f"✓ DataChannel opened: {channel.label}")
        print(f"  DataChannel state: {channel.readyState}")
        
        @channel.on("open")
        def on_open():
            print("✓ DataChannel fully open and ready to send")
            
        @channel.on("close")
        def on_close():
            print("✗ DataChannel closed")
            
        @channel.on("message")
        def on_message(message):
            print(f"← Received message from client: {message}")
        
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            
    @pc.on("track")
    def on_track(track):
        print(f"✓ Track received: {track.kind}")
        if track.kind == "video":
            # Pass the holder dict instead of the channel directly
            track_consumer = InferenceVideoTrack(track, data_channel_holder)
            pc.addTrack(track_consumer)
            print(f"✓ Added inference track with datachannel holder")
            
            @track.on("ended")
            async def on_ended():
                print("Track ended")
                
    try:
        offer_desc = RTCSessionDescription(sdp=sdp.sdp, type=sdp.type)
        await pc.setRemoteDescription(offer_desc)
        print("✓ Remote description set")
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("✓ Local description set")
        print("✓ WebRTC handshake complete")
        
    except Exception as e:
        print(f"✗ Error during WebRTC setup: {e}")
        pcs.discard(pc)
        raise
    
    async def cleanup():
        await asyncio.sleep(60*30)
        if pc in pcs:
            print(f"Cleaning up peer connection after timeout")
            try:
                await pc.close()
            except Exception:
                pass
            pcs.discard(pc)
    asyncio.ensure_future(cleanup())
    
    print("=" * 50)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/")
def index():
    return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
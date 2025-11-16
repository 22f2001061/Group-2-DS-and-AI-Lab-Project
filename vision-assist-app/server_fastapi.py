# server_fastapi.py
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
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# try local pyttsx3 (faster / offline). fallback to gTTS if pyttsx3 isn't available.
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

# YOLO
from ultralytics import YOLO

# ----------------------
# Configuration (env overrides)
# ----------------------
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo11s.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")  # "cuda" or "cpu"
TTS_CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

DEFAULT_PERSON_HEIGHT_M = float(os.environ.get("DEFAULT_PERSON_HEIGHT_M", 1.7))
FOCAL_LENGTH_PX = float(os.environ.get("FOCAL_LENGTH_PX", 1000.0))
ALERT_DISTANCE_PERSON_M = float(os.environ.get("ALERT_DISTANCE_PERSON_M", 2.5))
ALERT_DISTANCE_OBJECT_M = float(os.environ.get("ALERT_DISTANCE_OBJECT_M", 5.0))

ALERT_CLASS_COOLDOWN_SEC = float(os.environ.get("ALERT_CLASS_COOLDOWN_SEC", 6.0))
ALERT_REPEAT_DELAY_SEC = float(os.environ.get("ALERT_REPEAT_DELAY_SEC", 10.0))
ALERT_GLOBAL_COOLDOWN_SEC = float(os.environ.get("ALERT_GLOBAL_COOLDOWN_SEC", 0.5))

THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# ----------------------
# Helpers
# ----------------------
def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def text_to_audio_bytes_with_cache(text: str) -> Tuple[bytes, str]:
    """
    Produce WAV bytes (browser-friendly). Uses pyttsx3 if available (local/offline).
    Falls back to gTTS (MP3) if pyttsx3 isn't available.
    Returns (bytes, mime_type)
    """
    key = _text_hash(text)
    cache_wav = os.path.join(TTS_CACHE_DIR, f"{key}.wav")
    cache_mp3 = os.path.join(TTS_CACHE_DIR, f"{key}.mp3")

    # prefer pyttsx3 -> WAV
    if HAS_PYTTSX3:
        if os.path.exists(cache_wav):
            with open(cache_wav, "rb") as f:
                return f.read(), "audio/wav"
        try:
            engine = pyttsx3.init()
            # use temporary file because pyttsx3.save_to_file requires a filename
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            engine.save_to_file(text, tmp)
            engine.runAndWait()
            with open(tmp, "rb") as f:
                data = f.read()
            # cache
            with open(cache_wav, "wb") as f:
                f.write(data)
            try:
                os.remove(tmp)
            except Exception:
                pass
            return data, "audio/wav"
        except Exception:
            # proceed to fallback
            pass

    # fallback gTTS -> MP3
    if HAS_GTTS:
        if os.path.exists(cache_mp3):
            with open(cache_mp3, "rb") as f:
                return f.read(), "audio/mpeg"
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            mp3 = buf.getvalue()
            with open(cache_mp3, "wb") as f:
                f.write(mp3)
            return mp3, "audio/mpeg"
        except Exception:
            pass

    # final fallback: empty bytes
    return b"", "application/octet-stream"

def estimate_distance_px(box_h_px: float, object_real_h_m: float, focal_px: float = FOCAL_LENGTH_PX) -> float:
    if box_h_px <= 0 or math.isinf(box_h_px):
        return float("inf")
    return (object_real_h_m * focal_px) / float(box_h_px)

# ----------------------
# Alert manager
# ----------------------
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

# ----------------------
# Safe extraction of boxes, cls, ids
# ----------------------
def extract_boxes_cls_ids_from_result(result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Accepts ultralytics result and returns (xyxy_np, cls_np, id_np) or (None, None, None)
    Highly defensive: handles Torch tensors, numpy arrays, lists, or empty objects.
    """
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return None, None, None

    # xyxy
    xyxy_tensor = getattr(boxes_obj, "xyxy", None)
    if xyxy_tensor is None:
        return None, None, None
    try:
        # try torch-like
        xyxy = xyxy_tensor.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array(xyxy_tensor)
        except Exception:
            xyxy = None
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

    # track ids (may be None)
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

# ----------------------
# Load YOLO
# ----------------------
print(f"Loading YOLO model from {MODEL_PATH} ...")
yolo = YOLO(MODEL_PATH)
try:
    yolo.to(DEVICE)
except Exception:
    pass

# find person class id
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
# FastAPI app and simple index
# ----------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <html>
      <head><title>Inference Modes</title></head>
      <body>
        <h2>Inference demo</h2>
        <ul>
          <li><a href="/frame-inference">Frame Inference (single image)</a></li>
          <li><a href="/video-inference">Video Inference (SSE streaming)</a></li>
          <li><a href="/stream-inference">Stream Inference (WebRTC/camera -> websocket)</a></li>
        </ul>
      </body>
    </html>
    """
    return HTMLResponse(html)

# ----------------------
# Frame inference route (single image + immediate response)
# ----------------------
@app.get("/frame-inference", response_class=HTMLResponse)
async def frame_inference_page():
    html = """
    <!doctype html>
    <html>
      <head><title>Frame Inference</title></head>
      <body>
        <h3>Frame Inference (upload one image)</h3>
        <input id="file" type="file" accept="image/*"/>
        <button id="send">Send</button>
        <div id="out"></div>
        <script>
          document.getElementById("send").onclick = async () => {
            const f = document.getElementById("file").files[0];
            if (!f) return alert("Choose an image");
            const fd = new FormData();
            fd.append("file", f);
            const resp = await fetch("/frame_infer", { method: "POST", body: fd });
            if(!resp.ok) {
              const txt = await resp.text();
              alert("Error: " + txt);
              return;
            }
            const j = await resp.json();
            document.getElementById("out").innerText = j.text || "no text";
            // play audio
            if (j.audio_b64) {
              const mime = j.mime || "audio/wav";
              const ab = Uint8Array.from(atob(j.audio_b64), c => c.charCodeAt(0)).buffer;
              const blob = new Blob([ab], { type: mime });
              const url = URL.createObjectURL(blob);
              const audio = new Audio(url);
              audio.play();
            }
            // display image with bboxes if provided
            if (j.image_b64) {
              const imgEl = document.createElement("img");
              imgEl.src = "data:image/png;base64," + j.image_b64;
              imgEl.style.maxWidth = "480px";
              document.getElementById("out").appendChild(document.createElement("br"));
              document.getElementById("out").appendChild(imgEl);
            }
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/frame_infer")
async def frame_infer(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid_image")

    # run YOLO inference (no persistent track)
    try:
        results = yolo.predict(frame, conf=0.35, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detector_error:{e}")

    if not results or len(results) == 0:
        return JSONResponse({"text": "No objects detected.", "audio_b64": None, "mime": None})

    res0 = results[0]
    xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
    if xyxy is None:
        return JSONResponse({"text": "No boxes.", "audio_b64": None, "mime": None})

    h, w = frame.shape[:2]
    alerts = []
    for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
        try:
            x1, y1, x2, y2 = map(int, box[:4])
        except Exception:
            print(f"Invalid box: {box}")
            continue
        box_h_px = max(1, y2 - y1)
        object_real_h = DEFAULT_PERSON_HEIGHT_M if int(cls_id) == PERSON_CLASS_ID else DEFAULT_PERSON_HEIGHT_M
        dist = estimate_distance_px(box_h_px, object_real_h)
        cx = (x1 + x2) // 2
        rel = (cx - (w / 2)) / (w / 2)
        direction = "Right" if rel > 0.25 else ("Left" if rel < -0.25 else "Ahead")
        class_name = yolo.names.get(int(cls_id), f"Class{int(cls_id)}")
        is_person = (int(cls_id) == PERSON_CLASS_ID)
        is_close_person = is_person and (dist < ALERT_DISTANCE_PERSON_M)
        is_close_object = (not is_person) and (dist < ALERT_DISTANCE_OBJECT_M)
        # if is_close_person or is_close_object:
        alerts.append({"class_name": str(class_name), "distance": float(dist), "direction": direction,
                           "box": (x1, y1, x2, y2)})

    if not alerts:
        text = "No close objects detected."
        return JSONResponse({"text": text, "audio_b64": None, "mime": None})

    # pick closest
    alerts.sort(key=lambda x: x["distance"])
    chosen = alerts[0]
    text = f"Caution: {chosen['class_name']}, about {chosen['distance']:.1f} meters, {chosen['direction']}."
    # TTS in executor
    loop = asyncio.get_running_loop()
    audio_bytes, mime = await loop.run_in_executor(THREAD_POOL, text_to_audio_bytes_with_cache, text)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

    # optionally draw bbox on image and return as PNG b64
    x1, y1, x2, y2 = chosen["box"]
    vis = frame.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"{chosen['class_name']} {chosen['distance']:.1f}m", (x1, max(0, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    _, png = cv2.imencode(".png", vis)
    img_b64 = base64.b64encode(png.tobytes()).decode("utf-8")

    return JSONResponse({"text": text, "audio_b64": audio_b64, "mime": mime, "image_b64": img_b64})

# ----------------------
# Video inference page + upload endpoint (SSE)
# ----------------------
@app.get("/video-inference", response_class=HTMLResponse)
async def video_inference_page():
    html = """
    <!doctype html>
    <html>
      <head><title>Video Inference</title></head>
      <body>
        <h3>Video Inference (upload video)</h3>
        <input id="file" type="file" accept="video/*"/>
        <button id="send">Upload & Stream</button>
        <div id="log"></div>
        <script>
          document.getElementById("send").onclick = async () => {
            const f = document.getElementById("file").files[0];
            if (!f) return alert("Choose a video");
            const fd = new FormData();
            fd.append("file", f);
            // upload and get back an SSE url
            const resp = await fetch("/upload_video", { method: "POST", body: fd });
            if (!resp.ok) {
              const txt = await resp.text();
              alert("Upload failed: " + txt);
              return;
            }
            // We will open an EventSource to the returned URL, but since /upload_video returns streaming response directly,
            // we can use it via fetch readable stream.
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";
            const log = document.getElementById("log");
            function handleEventChunk(chunk) {
              buf += chunk;
              while (true) {
                const sep = "\\n\\n";
                const idx = buf.indexOf(sep);
                if (idx === -1) break;
                const evt = buf.slice(0, idx);
                buf = buf.slice(idx + sep.length);
                // parse lines
                const lines = evt.split("\\n").filter(Boolean);
                let ev = null;
                let data = null;
                for (const line of lines) {
                  if (line.startsWith("event:")) ev = line.slice(6).trim();
                  if (line.startsWith("data:")) {
                    try { data = JSON.parse(line.slice(5).trim()); } catch(e) { data = line.slice(5).trim(); }
                  }
                }
                if (ev === "alert" && data) {
                  log.innerText += "\\nALERT: " + (data.text || JSON.stringify(data));
                  if (data.audio_b64) {
                    const ab = Uint8Array.from(atob(data.audio_b64), c => c.charCodeAt(0)).buffer;
                    const mime = data.mime || "audio/wav";
                    const blob = new Blob([ab], { type: mime });
                    const url = URL.createObjectURL(blob);
                    const audio = new Audio(url);
                    audio.play();
                  }
                } else if (ev === "done") {
                  log.innerText += "\\nDONE";
                } else if (ev === "error") {
                  log.innerText += "\\nERROR: " + JSON.stringify(data);
                }
              }
            }
            // read stream chunks
            (async () => {
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const s = decoder.decode(value, { stream: true });
                handleEventChunk(s);
              }
            })();
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video upload and returns SSE streaming events (text/event-stream).
    Each event has:
      event: alert
      data: {"text":..., "audio_b64":..., "timestamp_ms":...}
    """
    contents = await file.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    async def generator():
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield f"event: error\ndata: {json.dumps({'message':'could_not_open_video'})}\n\n"
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return

        alert_mgr = AlertManager()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # run yolo track (persist=True to get ids where available)
                try:
                    results = yolo.track(frame, conf=0.35, persist=True, verbose=False)
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'message':'detector_error','detail':str(e)})}\n\n"
                    continue

                if not results or len(results) == 0:
                    continue
                res0 = results[0]
                xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
                if xyxy is None:
                    continue

                h, w = frame.shape[:2]
                potential = []
                timestamp_ms = int(time.time() * 1000)
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
                    direction = "Right" if rel > 0.25 else ("Left" if rel < -0.25 else "Ahead")
                    class_name = yolo.names.get(int(cls_id), f"Class{int(cls_id)}")
                    is_person = (int(cls_id) == PERSON_CLASS_ID)
                    is_close_person = is_person and (dist < ALERT_DISTANCE_PERSON_M)
                    is_close_object = (not is_person) and (dist < ALERT_DISTANCE_OBJECT_M)
                    if (is_close_person or is_close_object) and alert_mgr.can_alert(int(track_id), int(cls_id), time.time()):
                        potential.append({"track_id": int(track_id), "class_id": int(cls_id),
                                          "class_name": str(class_name), "distance": float(dist),
                                          "direction": direction})
                if not potential:
                    continue
                # prioritize persons then distance
                potential.sort(key=lambda x: (0 if x["class_id"] == PERSON_CLASS_ID else 1, x["distance"]))
                chosen = potential[0]
                text = f"Caution: {chosen['class_name']}, about {chosen['distance']:.1f} meters, {chosen['direction']}."
                timestamp_ms = int(time.time() * 1000)
                alert_mgr.register_alert(chosen['track_id'], chosen['class_id'], timestamp_ms)
                loop = asyncio.get_running_loop()
                try:
                    audio_bytes, mime = await loop.run_in_executor(THREAD_POOL, text_to_audio_bytes_with_cache, text)
                except Exception:
                    audio_bytes, mime = b"", "application/octet-stream"
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None
                payload = {"text": text, "audio_b64": audio_b64, "mime": mime, "timestamp_ms": timestamp_ms}
                yield f"event: alert\ndata: {json.dumps(payload)}\n\n"

            yield f"event: done\ndata: {json.dumps({'message':'finished'})}\n\n"
        finally:
            cap.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return StreamingResponse(generator(), media_type="text/event-stream")

# ----------------------
# Stream inference (WebSocket real-time camera)
# ----------------------
@app.get("/stream-inference", response_class=HTMLResponse)
async def stream_inference_page():
    html = """
    <!doctype html>
    <html>
      <head><title>Stream Inference</title></head>
      <body>
        <h3>Stream Inference (camera -> websocket)</h3>
        <video id="v" autoplay playsinline width="480" style="border:1px solid #ccc"></video><br/>
        <button id="start">Start</button>
        <button id="stop">Stop</button>
        <div id="logs"></div>
        <script>
          let ws;
          let sender;
          let running = false;
          document.getElementById("start").onclick = async () => {
            if (running) return;
            running = true;
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            const v = document.getElementById("v");
            v.srcObject = stream;
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
            ws.onopen = () => {
              document.getElementById("logs").innerText += "\\nWS open";
              sender = setInterval(async () => {
                canvas.width = v.videoWidth || 480;
                canvas.height = v.videoHeight || 360;
                ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
                // send as JSON
                ws.send(JSON.stringify({ type: "frame", b64: dataUrl }));
              }, 300); // send ~3 fps (tuneable)
            };
            ws.onmessage = (ev) => {
              try {
                const obj = JSON.parse(ev.data);
                if (obj.type === "audio" && obj.audio_b64) {
                  const ab = Uint8Array.from(atob(obj.audio_b64), c => c.charCodeAt(0)).buffer;
                  const blob = new Blob([ab], { type: "audio/wav" });
                  const url = URL.createObjectURL(blob);
                  const audio = new Audio(url);
                  audio.play();
                  document.getElementById("logs").innerText += "\\n" + (obj.text || "Audio received");
                }
              } catch(e) {
                console.error(e);
              }
            };
            ws.onclose = () => document.getElementById("logs").innerText += "\\nWS closed";
            ws.onerror = (e) => document.getElementById("logs").innerText += "\\nWS error";
          };
          document.getElementById("stop").onclick = () => {
            running = false;
            if (sender) clearInterval(sender);
            if (ws) ws.close();
            const v = document.getElementById("v");
            if (v.srcObject) {
              v.srcObject.getTracks().forEach(t => t.stop());
              v.srcObject = null;
            }
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected (websocket camera).")
    alert_mgr = AlertManager()
    try:
        while True:
            msg = await ws.receive_text()
            # expect JSON with dataURL in .b64
            try:
                payload = json.loads(msg)
                data_url = payload.get("b64") or msg
            except Exception:
                data_url = msg

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

            # inference with tracking
            try:
                results = yolo.track(frame, conf=0.35, persist=True, verbose=False)
            except Exception as e:
                await ws.send_json({"type": "error", "message": "detector_error", "detail": str(e)})
                continue

            if not results or len(results) == 0:
                # nothing detected
                continue

            res0 = results[0]
            xyxy, cls_ids, track_ids = extract_boxes_cls_ids_from_result(res0)
            if xyxy is None:
                continue

            h, w = frame.shape[:2]
            potential = []
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
                direction = "Right" if rel > 0.25 else ("Left" if rel < -0.25 else "Ahead")
                class_name = yolo.names.get(int(cls_id), f"Class{int(cls_id)}")
                is_person = (int(cls_id) == PERSON_CLASS_ID)
                is_close_person = is_person and (dist < ALERT_DISTANCE_PERSON_M)
                is_close_object = (not is_person) and (dist < ALERT_DISTANCE_OBJECT_M)
                if (is_close_person or is_close_object) and alert_mgr.can_alert(int(track_id), int(cls_id), time.time()):
                    potential.append({"track_id": int(track_id), "class_id": int(cls_id),
                                      "class_name": str(class_name), "distance": float(dist),
                                      "direction": direction})
            if not potential:
                continue
            potential.sort(key=lambda x: (0 if x["class_id"] == PERSON_CLASS_ID else 1, x["distance"]))
            chosen = potential[0]
            text = f"Caution: {chosen['class_name']}, about {chosen['distance']:.1f} meters, {chosen['direction']}."
            timestamp_ms = int(time.time() * 1000)
            alert_mgr.register_alert(chosen['track_id'], chosen['class_id'], timestamp_ms)
            loop = asyncio.get_running_loop()
            try:
                audio_bytes, mime = await loop.run_in_executor(THREAD_POOL, text_to_audio_bytes_with_cache, text)
            except Exception:
                audio_bytes, mime = b"", "application/octet-stream"
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None
            await ws.send_json({"type": "audio", "text": text, "audio_b64": audio_b64, "mime": mime, "timestamp_ms": timestamp_ms})

    except Exception as e:
        print("WebSocket ended.", e)
    finally:
        print("WebSocket client disconnected.")

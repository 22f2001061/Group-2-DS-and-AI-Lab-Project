import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from inference.yolo_loader import yolo, PERSON_CLASS_ID
from webrtc.offer_handler import offer
from config.settings import STATIC_DIR


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.post("/offer")(offer)


@app.get("/")
def index():
    return HTMLResponse(open(f"{STATIC_DIR}/index.html", "r", encoding="utf-8").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

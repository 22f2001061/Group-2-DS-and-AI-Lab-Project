import gradio as gr
from app.detector import Detector
from app.tracker import ObjectTracker
from app.distance import DistanceEstimator
from app.alert_manager import AlertManager
from app.tts import TTS
from app.pipeline import run_navigation_pipeline

detector = Detector('yolov8n.pt')
tracker = ObjectTracker()
distance = DistanceEstimator(focal_length_px=950, default_height_m=1.7)
alert_mgr = AlertManager(per_class_cd=6, repeat_delay=10)
tts = TTS()

def process(video):
    output = "output_result.mp4"
    run_navigation_pipeline(video.name, output, detector, tracker, distance, alert_mgr, tts)
    return output

demo = gr.Interface(
    fn=process,
    inputs=gr.Video(label="Upload video"),
    outputs=gr.Video(label="Processed video with navigation cues"),
    title="Blind Navigation Assistant",
    description="Object detection + spoken navigation cues for low-vision users."
)

if __name__ == "__main__":
    demo.launch(share=True)

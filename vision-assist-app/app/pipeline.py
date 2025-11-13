import cv2, os, time, tempfile, subprocess
from pydub import AudioSegment

def run_navigation_pipeline(video_path, output_path,
                            detector, tracker, distance_estimator,
                            alert_manager, tts_engine,
                            conf_thresh=0.4, target_classes=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    total_alert_audio = AudioSegment.silent(duration=0)
    alert_manager.alert_log.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp_ms = frame_idx * 1000 / fps

        results = detector.detect(frame)
        active_ids = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.id is None: continue
                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                conf = float(box.conf.item())

                if target_classes and class_id not in target_classes:
                    continue
                if conf < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                h_box = y2 - y1
                distance_m = distance_estimator.estimate(h_box, class_id)

                tracker.update(track_id, cx, cy)
                direction, motion = tracker.get_motion_direction(track_id, width)

                text = f"{results[0].names[class_id]} {direction}, {motion}, {distance_m:.1f} meters"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if alert_manager.can_alert(track_id, class_id):
                    alert_manager.register_alert(track_id, class_id, text, timestamp_ms)
                    total_alert_audio += tts_engine.synthesize(text)
                active_ids.append(track_id)

        tracker.remove_missing(active_ids)
        writer.write(frame)

    cap.release()
    writer.release()

    # Mix video + audio
    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    total_alert_audio.export(audio_temp, format="mp3")

    # combine using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", temp_video, "-i", audio_temp,
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(temp_video)
    os.remove(audio_temp)
    return output_path

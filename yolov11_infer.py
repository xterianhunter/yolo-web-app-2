import cv2
import os
from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolov11.pt")
FRAME_DIR = "static/frames"
os.makedirs(FRAME_DIR, exist_ok=True)

stop_streaming = False

def generate_frames():
    global stop_streaming
    cap = cv2.VideoCapture(0)

    while True:
        if stop_streaming:
            break

        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        # Save frame
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_path = os.path.join(FRAME_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(save_path, annotated)

        # Encode frame for browser
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def stop_detection():
    global stop_streaming
    stop_streaming = True

def start_detection():
    global stop_streaming
    stop_streaming = False

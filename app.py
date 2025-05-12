import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import threading

app = Flask(__name__)
app.secret_key = "realtimekey"

# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = YOLO("yolov11.pt")

# Global flag to manage real-time detection
detecting = False
camera_thread = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global detecting
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        if not detecting:
            break

        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLO object detection on the frame
        results = model(frame)
        annotated_frame = results[0].plot()  # Annotate the frame with detection results

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    global detecting
    uploaded_image = session.get("uploaded_image", None)
    return render_template('index.html', detecting=detecting, uploaded_image=uploaded_image)

@app.route('/start-realtime', methods=['POST'])
def start_realtime():
    global detecting, camera_thread
    detecting = True
    # Start the camera stream in a separate thread
    camera_thread = threading.Thread(target=generate_frames)
    camera_thread.start()
    flash("✅ Real-time detection started.")
    return redirect(url_for('index'))

@app.route('/stop-realtime', methods=['POST'])
def stop_realtime():
    global detecting, camera_thread
    detecting = False
    if camera_thread is not None:
        camera_thread.join()  # Ensure the thread is properly closed
    flash("🛑 Real-time detection stopped.")
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash("No file part")
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Perform object detection
        img = cv2.imread(file_path)
        results = model(img)
        annotated_image = results[0].plot()  # Get annotated image

        # Save the annotated image
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)

        # Store the filename of the uploaded image for displaying in the frontend
        session["uploaded_image"] = annotated_filename
        flash("✅ Image uploaded and detected successfully.")
        return redirect(url_for('index'))
    else:
        flash("Invalid file type. Please upload a .jpg, .jpeg, .png, or .gif file.")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

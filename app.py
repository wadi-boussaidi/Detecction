from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
from collections import OrderedDict
from threading import Thread
import time

app = Flask(__name__)

# Configuration
MODEL_PATH = "animal_classifier.h5"
CLASS_LABELS = ["elefante", "farfalla", "mucca", "pecora", "scoiattolo", "wadii"]
CAMERA_INDEX = 0
MAX_DISAPPEARED = 10
MIN_CONTOUR_AREA = 2000
THRESHOLD_VALUE = 25
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 80

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize camera
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Background subtractor for moving object detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Object Tracker (Centroid Tracking Algorithm)
class ObjectTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        input_rects = []

        for (x, y, w, h) in rects:
            c_x = int(x + w / 2)
            c_y = int(y + h / 2)
            input_centroids.append((c_x, c_y))
            input_rects.append((x, y, w, h))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], input_rects[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_rects[col])

        return self.objects

tracker = ObjectTracker()

def preprocess_frame(frame):
    """Preprocess the frame for model prediction."""
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def detect_moving_objects(frame):
    """Detect moving objects using background subtraction."""
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

def generate_frames():
    """Generate frames with object tracking and classification."""
    while True:
        start_time = time.time()
        success, frame = camera.read()
        if not success:
            break

        # Detect moving objects
        bounding_boxes = detect_moving_objects(frame)

        # Update tracker with detected objects
        tracked_objects = tracker.update(bounding_boxes)

        # Draw bounding boxes and labels
        for object_id, (centroid, bbox) in tracked_objects.items():
            (x, y, w, h) = bbox

            # Draw bounding box and object ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Classify the object in the bounding box
            focus_frame = frame[y:y + h, x:x + w]
            input_frame = preprocess_frame(focus_frame)
            predictions = model.predict(input_frame)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Display classification result
            label = f"{predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        # Limit FPS
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 / FPS - elapsed_time))

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Stream video feed with object tracking and classification."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
import cv2
import subprocess
import json
import numpy as np
import torch
from collections import Counter
from ultralytics import YOLO

# YouTube live stream URL
YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=C3FyLUnHHySL10Ou"


def get_stream_url(url):
    """Extracts the direct stream URL using yt-dlp."""
    cmd = ["yt-dlp", "-j", "-f", "best[ext=mp4]", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return info["url"]


# Initialize stream capture
stream_url = get_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("Failed to open video stream")

# Get stream FPS for time calculations
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# Use GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Running on DEVICE: {device}")

# Load YOLOv8m (Medium) - Better balance between speed and accuracy
model = YOLO("yolov8m.pt")
model.to(device)

# Detection threshold
CONF_THRESH = 0.4

# Target classes from COCO dataset (added person as per previous request)
CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Calibration: meters per pixel (adjusted for ~60 km/h average)
meter_per_pixel = 0.025

# Tracking dictionaries
prev_centers = {}  # Store previous frame coordinates: {track_id: (x, y)}
speed_history = {}  # Store speed history for smoothing: {track_id: [speeds]}
seen_track_ids = set()  # Track unique objects for the counter
total_counter = Counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (using ByteTrack)
    results = model.track(
        frame,
        conf=CONF_THRESH,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    r = results[0]

    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()

        new_centers = {}

        for i in range(len(xyxy)):
            class_id = int(class_ids[i])
            if class_id not in CLASSES:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = int(track_ids[i])
            class_name = CLASSES[class_id]

            new_centers[tid] = (cx, cy)

            # Update unique object counter
            if tid not in seen_track_ids:
                seen_track_ids.add(tid)
                total_counter[class_name] += 1

            # Speed calculation logic
            current_speed = 0.0
            if tid in prev_centers:
                # Calculate pixel displacement
                dx = cx - prev_centers[tid][0]
                dy = cy - prev_centers[tid][1]
                dist_px = np.sqrt(dx ** 2 + dy ** 2)

                # Convert distance to meters and speed to km/h
                dist_m = dist_px * meter_per_pixel
                instant_speed = dist_m * fps * 3.6

                # Moving average smoothing (last 12 frames)
                if tid not in speed_history:
                    speed_history[tid] = []
                speed_history[tid].append(instant_speed)
                if len(speed_history[tid]) > 12:
                    speed_history[tid].pop(0)

                current_speed = np.mean(speed_history[tid])

            # Visuals: Bounding box and label
            color = (0, 255, 0) if class_id != 0 else (255, 0, 0)  # Green for vehicles, Blue for people
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            speed_label = f"{class_name} | {current_speed:.1f} km/h"
            cv2.putText(
                frame, speed_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Update tracking data
        prev_centers = new_centers

    # UI: Counter Sidebar
    h_box = 40 + 26 * len(total_counter)
    cv2.rectangle(frame, (5, 5), (250, h_box), (0, 0, 0), -1)

    y_text = 30
    cv2.putText(frame, "LIVE STATISTICS", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_text += 28

    for obj_type, count in total_counter.items():
        cv2.putText(frame, f"{obj_type.capitalize()}: {count}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_text += 24

    # Display result
    cv2.imshow("Traffic Analytics (YOLOv8m)", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
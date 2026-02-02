import cv2
import numpy as np
import shutil
import os


PROJECT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODUL_DIR = os.path.join(PROJECT_DIR, 'modul')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
PEOPLE_DIR = os.path.join(OUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUT_DIR, 'no_people')
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)
PROTOXTX_DIR = os.path.join(MODUL_DIR, 'MobileNetSSD_deploy.prototxt.txt')
MODUL_PATH = os.path.join(MODUL_DIR, 'MobileNetSSD_deploy.caffemodel')
net = cv2.dnn.readNetFromCaffe(PROTOXTX_DIR, MODUL_PATH)
CLASSES = [
"background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"
]
PERSON_CLASS_ID = CLASSES.index('person')
CONF_THRESHOLD = 0.1
def detect_person(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.0073, (300, 300), mean=(104, 117, 123))
    net.setInput(blob)
    detections = net.forward()
    best_conf = 0.0
    best_box = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]
        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2, y2)
    found = best_box is not None
    return found, best_box, best_conf

allowed_extensions = ('jpg', 'jpeg', 'png', '.bmp')
files = os.listdir(IMAGES_DIR)
count_people = 0
count_no_people = 0
for file in files:
    if not file.lower().endswith(allowed_extensions):
        continue
    in_path = os.path.join(IMAGES_DIR, file)
    img = cv2.imread(in_path)
    found, best_box, best_conf = detect_person(img)
    if found:
        out_path = os.path.join(PEOPLE_DIR, file)
        shutil.copy(in_path, out_path)
        count_people += 1

        boxed = img.copy()
        x1, y1, x2, y2 = best_box
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(boxed,f"{best_conf:.2f}",(x1, y1 + 10),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 2)

        boxed_path  = os.path.join(PEOPLE_DIR, "boxed_" + file)
        cv2.imwrite(boxed_path, boxed)
    else:
        count_no_people += 1
        out_path = os.path.join(NO_PEOPLE_DIR, file)
        shutil.copy(in_path, out_path)

    print(f'people: {count_people}, no_people: {count_no_people}')
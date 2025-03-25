
GPIO final.py
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import pyttsx3
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for buttons
BUTTON_O = 17
BUTTON_T = 27
BUTTON_D = 22
BUTTON_S = 23

# Set up GPIO pins
GPIO.setup(BUTTON_O, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_T, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_D, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_S, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1)

# Load YOLO model
model = YOLO('/home/pi/models/yolov5n.pt')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function for text detection using PaddleOC R
def detect_text(frame):
    results = ocr.ocr(frame, cls=True)
    detected_texts = []
    for line in results[0]:
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5:
            detected_texts.append(text)
    return " ".join(detected_texts)

# Function to estimate the distance to an object
def estimate_distance(box, frame_width):
    FOCAL_LENGTH = 615  # Example focal length
    KNOWN_WIDTH = 20    # Known width of the object in cm
    box_width_pixels = box[2] - box[0]
    if box_width_pixels == 0:
        return None
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width_pixels
    return distance

# Video capture from webcam
cap = cv2.VideoCapture(0)

mode = None  # Track the detection mode (object, text, distance)
detecting = False  # Track detection state

# Function to check GPIO button inputs
def check_buttons():
    global mode, detecting
    if GPIO.input(BUTTON_O) == GPIO.HIGH:
        mode = "object"
        detecting = True
    elif GPIO.input(BUTTON_T) == GPIO.HIGH:
        mode = "text"
        detecting = True
    elif GPIO.input(BUTTON_D) == GPIO.HIGH:
        mode = "distance"
        detecting = True
    elif GPIO.input(BUTTON_S) == GPIO.HIGH:
        detecting = False
        mode = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    try:
        # Check button inputs
        check_buttons()

        if mode == "object" and detecting:
            results = model(frame)  # YOLO inference
            detected_objects = []
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = result.names[int(cls)]
                    confidence = float(conf)
                    if confidence > 0.5:
                        detected_objects.append(label)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if detected_objects:
                top_objects = detected_objects[:5]
                speak(f"Detected objects: {', '.join(top_objects)}")

        elif mode == "text" and detecting:
            text = detect_text(frame)
            if text.strip():
                print("Detected Text:", text)
                speak(text)

        elif mode == "distance" and detecting:
            results = model(frame)  # YOLO inference
            distances = []
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = result.names[int(cls)]
                    confidence = float(conf)
                    if confidence > 0.5:
                        distance = estimate_distance((x1, y1, x2, y2), frame.shape[1])
                        if distance is not None:
                            distances.append(f"{label}: {distance:.2f} cm")

            if distances:
                top_distances = distances[:5]
                speak(f"Distances: {', '.join(top_distances)}")

    except Exception as e:
        print(f"Error occurred: {e}")
        speak("An error occurred")
        mode = None
        detecting = False

    # Display the frame
    cv2.imshow("Detection", frame)

    # Key presses to switch modes
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        mode = "object"
        detecting = True
    elif key == ord('t'):
        mode = "text"
        detecting = True
    elif key == ord('d'):
        mode = "distance"
        detecting = True
    elif key == ord('s'):
        detecting = False
        mode = None
    elif key == ord('q'):
        break

# Release resources
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()



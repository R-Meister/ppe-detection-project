import cv2
from ultralytics import YOLO
import os

# Suppress YOLO's verbose output
os.environ['YOLO_VERBOSE'] = 'False'

# Load your custom trained model
model = YOLO("runs/detect/train/weights/best.pt")  # adjust path if needed

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Class mapping
CLASS_NAMES = {1: 'Helmet'}  # only care about class 1

# Initialize status display variables
last_status = None
status_display_time = 0
STATUS_DISPLAY_DURATION = 30  # frames to display status

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection with verbose=False
    results = model(frame, verbose=False)
    
    # Check for helmet detection
    helmet_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 1:  # Only helmet
                helmet_detected = True
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Update status display
    current_status = "✅ Helmet Detected" if helmet_detected else "❌ No Helmet"
    if current_status != last_status:
        print(current_status)
        last_status = current_status
        status_display_time = STATUS_DISPLAY_DURATION

    # Display result
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import os

# Suppress YOLO's verbose output
os.environ['YOLO_VERBOSE'] = 'False'

# Load your custom trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Class mapping
CLASS_NAMES = {0: 'Head', 1: 'Helmet'}

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
    
    # Check for head and helmet detection
    head_detected = False
    helmet_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 0:  # Head
                head_detected = True
                # Draw red box for head
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif cls == 1:  # Helmet
                helmet_detected = True
                # Draw green box for helmet
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Update status display
    if helmet_detected:
        current_status = "✅ Helmet Detected"
    elif head_detected:
        current_status = "⚠️ Head Visible (No Helmet)"
    else:
        current_status = "❌ Head Detected"
        
    if current_status != last_status:
        print(current_status)
        last_status = current_status
        status_display_time = STATUS_DISPLAY_DURATION

    # Display result
    cv2.imshow("Head/Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

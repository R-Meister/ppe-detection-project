from ultralytics import YOLO
import cv2

# Load the YOLOv8 nano model (fastest, smallest)
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(source=frame, show=True)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

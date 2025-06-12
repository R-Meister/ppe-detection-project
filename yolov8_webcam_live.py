from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt if you prefer

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # update if path differs

# Open the webcam
cap = cv2.VideoCapture(0)

# Optional: Increase webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame, conf=0.25, verbose=False)

    # Draw detections
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if cls_id == 0:  # head
            print("⚠️ Head Detected!")
            cv2.putText(
                annotated_frame,
                "⚠️ Head Detected!",
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Webcam Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



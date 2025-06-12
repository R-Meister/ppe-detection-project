from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("runs/detect/train/weights/best.pt")

# Load test image
image = cv2.imread("test.jpg.png")

# Run detection
results = model(image, conf=0.25)

# Save annotated result
results[0].save(filename="output.jpg")

print("✅ Detection complete. Check 'output.jpg'")

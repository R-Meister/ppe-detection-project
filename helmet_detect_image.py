from ultralytics import YOLO
import cv2
import sys
import os

def detect_helmet(image_path):
    # Suppress YOLO's verbose output
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Load the model - using the trained model instead of base model
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Run detection with verbose=False to suppress output
    results = model(image, conf=0.25, verbose=False)
    
    # Check for helmet detection
    helmet_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Class 1 is helmet in our PPE detection model
            if box.cls == 1:  # Changed from 0 to 1 for helmet class
                helmet_detected = True
                break
    
    # Save annotated result
    output_path = "output.jpg"
    results[0].save(filename=output_path)
    
    return helmet_detected, output_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python helmet_detect_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    helmet_detected, output_path = detect_helmet(image_path)
    
    # Only print the final result
    if helmet_detected:
        print("✅ Helmet detected!")
    else:
        print("❌ No helmet detected")
    print(f"Annotated image saved as: {output_path}") 
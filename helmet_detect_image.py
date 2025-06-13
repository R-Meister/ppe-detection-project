from ultralytics import YOLO
import cv2
import sys
import os

def detect_head_helmet(image_path):
    # Suppress YOLO's verbose output
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Load the model - using the trained model instead of base model
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False, False, None
    
    # Run detection with verbose=False
    results = model(image, verbose=False)
    
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
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif cls == 1:  # Helmet
                helmet_detected = True
                # Draw green box for helmet
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save annotated result
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    
    return head_detected, helmet_detected, output_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python helmet_detect_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    head_detected, helmet_detected, output_path = detect_head_helmet(image_path)
    
    # Print detection results
    if helmet_detected:
        print("✅ Helmet Detected")
    elif head_detected:
        print("⚠️ Head Visible (No Helmet)")
    else:
        print("❌ No Head/Helmet Detected")
    print(f"Annotated image saved as: {output_path}") 
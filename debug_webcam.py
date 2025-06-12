import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

ret, frame = cap.read()
if ret:
    print("Frame captured successfully.")
    cv2.imshow('Test Frame', frame)
    cv2.waitKey(3000)  # Show window for 3 seconds
else:
    print("Error: Could not read frame from webcam.")

cap.release()
cv2.destroyAllWindows()

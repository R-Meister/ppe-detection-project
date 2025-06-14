# PPE Detection Project

This project implements Personal Protective Equipment (PPE) detection using YOLOv8, specifically focusing on helmet detection. The system can detect helmets in both images and live webcam feeds.

## Features
- Helmet detection in images
- Real-time helmet detection using webcam
- Uses YOLOv8 pre-trained model for accurate detection

## Usage

### Image Detection
To detect helmets in an image:
```bash
python helmet_detect_image.py <path_to_image>
```
The script will output whether a helmet was detected and save an annotated image as 'output.jpg'.

### Webcam Detection
To run real-time helmet detection using your webcam:
```bash
python helmet_detect_webcam.py
```

## Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- PyTorch

## Installation
```bash
pip install ultralytics opencv-python torch
```

## Model
The project uses the YOLOv8 pre-trained model (yolov8n.pt) for detection.

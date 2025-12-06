# License Plate Recognition (YOLOv8 + EasyOCR)

This project detects vehicle license plates in images/videos using a fine-tuned YOLOv8 model and then reads the plate text using EasyOCR.

## Overview

- Object detection: YOLOv8 trained on a custom license plate dataset.
- OCR: EasyOCR to extract text from detected plate regions.
- Post-processing: Regex-based filtering to keep only valid plate patterns.

## How It Works

1. YOLOv8 detects bounding boxes for license plates.
2. Each plate crop is passed to EasyOCR.
3. A regex pattern (e.g. `XX11XXX`) filters out invalid text candidates.
4. Final plate text and bounding boxes are drawn on the frame.

## Setup

git clone https://github.com/aryankd123/License_plate_recognition_cv.git
cd License_plate_recognition_cv

Create & activate env (optional but recommended)
python -m venv venv
source venv/bin/activate # on macOS/Linux

venv\Scripts\activate # on Windows
pip install -r requirements.txt

text

Make sure you have the YOLO weights in:

saved_models/license_plate_best.pt

text

## Usage

### Image

from ultralytics import YOLO
import easyocr

model = YOLO("saved_models/license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

results = model("test_image.jpg")

then crop plate boxes and run EasyOCR on each crop
text

### Video (OpenCV)

import cv2

cap = cv2.VideoCapture("path/to/your_video.mp4")
while True:
ret, frame = cap.read()
if not ret:
break


## Project Structure

- `number_plate.ipynb` – main notebook (training + inference pipeline)
- `saved_models/` – fine-tuned YOLOv8 weights

## Future Improvements

- Support multiple plate formats.
- Add tracking across frames (e.g. ByteTrack/DeepSORT).
- Web or GUI interface for live camera feeds.


# Traffic Violation Detection System

A computer vision system that detects illegal stopping in yellow boxes and zebra crossings, reads license plates, and stores violation records in MongoDB.

## Features

- Real-time detection of vehicles illegally stopping in:
  - Yellow box junctions
  - Zebra crossings
- License plate recognition using EasyOCR
- Violation recording with:
  - Timestamp
  - Duration of violation
  - License plate number (when readable)
  - Snapshot image
  - Video clip of the violation
- Data storage in MongoDB
- Organized file storage for media

## Prerequisites

- Python 3.7+
- MongoDB server (local or remote)
- Camera or video source

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/traffic-violation-detector.git
   cd traffic-violation-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLO weights and config:
   ```bash
   wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
   ```

4. Create a .env file for environment variables if needed (modify config/settings.py directly for this simple setup)

## Configuration

Modify the configuration in config/settings.py:

- MongoDB connection settings
- Storage paths and limits
- Detection parameters (minimum stop time, confidence thresholds, etc.)
- License plate recognition settings

## Usage

Run the system:
```bash
   python main.py
```

Command line arguments:

- Use 0 for webcam or video file path/RTSP stream URL
- Press 'q' to quit

## Database Schema

Violations are stored in MongoDB with the following structure:
```json
   {
  "_id": "violation_id",
  "timestamp": ISODate,
  "license_plate": "String",
  "violation_type": "yellow_box|zebra_crossing",
  "location": [x, y, w, h],
  "duration": seconds,
  "image_path": "String",
  "video_path": "String",
  "status": "pending|processed|rejected"
}
```

## File Storage

- Images are saved in storage/images/
- Video clips are saved in storage/videos/
- The system automatically manages storage space and deletes oldest files when limit is reached

## Customization

1. Detection Zones:
   - Adjust color ranges for yellow box detection in config/settings.py
   - Modify zebra crossing detection parameters
2. License Plate Recognition:
   - Tune preprocessing in license_plate_recognizer.py
   - Change OCR reader configuration for your region
3. Vehicle Detection:
   - Replace YOLO with another model if needed
   - Adjust vehicle class IDs for your use case

## Limitations

- License plate recognition accuracy depends on:
   - Image quality
   - Lighting conditions
   - Plate visibility
- Zebra crossing detection is simplified and may need improvement for real-world use
- Performance depends on hardware (consider GPU acceleration for better performance)

   

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# MongoDB settings
MONGO_CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'db_name': 'traffic_violations',
    'collection': 'violations'
}

# File storage settings
STORAGE_CONFIG = {
    'images_dir': os.path.join(BASE_DIR, 'storage/images'),
    'videos_dir': os.path.join(BASE_DIR, 'storage/videos'),
    'max_storage_mb': 1024  # 1GB max storage
}

# Detection settings
DETECTION_CONFIG = {
    'min_stop_time': 3,  # seconds to consider as violation
    'confidence_threshold': 0.7,
    'yellow_box_color_range': ([20, 100, 100], [30, 255, 255]),  # HSV range
    'zebra_crossing_contour_area': 5000  # min area to consider as zebra crossing
}

# License plate settings
LP_CONFIG = {
    'min_width': 80,
    'min_height': 30,
    'max_width': 300,
    'max_height': 100,
    'contrast_enhance': 1.5,
    'brightness_enhance': 10
}
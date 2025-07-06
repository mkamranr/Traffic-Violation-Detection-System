import cv2
import time
from datetime import datetime
from collections import deque

from config.settings import DETECTION_CONFIG
from database.db_handler import MongoDBHandler
from detectors.violation_detector import ViolationDetector
from detectors.license_plate_recognizer import LicensePlateRecognizer
from utils.file_handler import FileHandler
from utils.helpers import draw_violation_info, draw_detection_zones

class TrafficViolationSystem:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.running = False
        
        # Initialize components
        self.violation_detector = ViolationDetector()
        self.lp_recognizer = LicensePlateRecognizer()
        self.db_handler = MongoDBHandler()
        self.file_handler = FileHandler()
        
        # Buffer for storing frames when violation occurs
        self.violation_frames = deque(maxlen=100)  # Store up to 100 frames
        
        # Track current violations
        self.current_violations = {}
    
    def start(self):
        """Start the violation detection system"""
        self.running = True
        print("Traffic violation detection system started")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Store frame in buffer
            self.violation_frames.append(frame.copy())
            
            # Detect vehicles
            vehicles = self.violation_detector.detect_vehicles(frame)
            
            # Detect restricted zones
            yellow_boxes = self.violation_detector.detect_yellow_boxes(frame)
            zebra_crossings = self.violation_detector.detect_zebra_crossings(frame)
            
            # Check for violations
            violations = self.violation_detector.check_violations(
                frame, vehicles, yellow_boxes, zebra_crossings
            )
            
            # Process violations
            for violation in violations:
                self._process_violation(violation, frame)
            
            # Draw detection zones
            frame = draw_detection_zones(frame, yellow_boxes, zebra_crossings)
            
            # Display frame
            cv2.imshow('Traffic Violation Detection', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check storage limits periodically
            if int(time.time()) % 60 == 0:  # Every minute
                self.file_handler.check_storage()
        
        # Clean up
        self.stop()
    
    def _process_violation(self, violation, frame):
        """Process a detected violation"""
        vehicle_id = violation['vehicle_id']
        
        if vehicle_id not in self.current_violations:
            # New violation
            print(f"New {violation['type']} violation detected!")
            
            # Get license plate
            license_plate, _ = self.lp_recognizer.recognize_from_frame(
                frame, violation['vehicle']['bbox']
            )
            
            if license_plate:
                print(f"License plate detected: {license_plate}")
            else:
                print("License plate not recognized")
                license_plate = "UNKNOWN"
            
            # Save violation data
            violation_data = {
                'violation_type': violation['type'],
                'duration': violation['duration'],
                'location': violation['location'],
                'license_plate': license_plate
            }
            
            # Save image
            image_path = self.file_handler.save_violation_image(frame, vehicle_id)
            violation_data['image_path'] = image_path
            
            # Save video clip (last 3 seconds)
            clip_frames = list(self.violation_frames)[-60:]  # Assuming 20fps, 3 seconds
            video_path = self.file_handler.save_violation_video(clip_frames, vehicle_id)
            violation_data['video_path'] = video_path
            
            # Store in database
            violation_id = self.db_handler.create_violation_record(violation_data)
            self.current_violations[vehicle_id] = {
                'violation_id': violation_id,
                'start_time': time.time()
            }
            
            print(f"Violation recorded with ID: {violation_id}")
        
        # Draw violation info on frame
        license_plate = self.current_violations.get(vehicle_id, {}).get('license_plate', 'UNKNOWN')
        frame = draw_violation_info(frame, violation, license_plate)
    
    def stop(self):
        """Stop the violation detection system"""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.db_handler.close_connection()
        print("System stopped")

if __name__ == "__main__":
    # For testing with webcam
    # system = TrafficViolationSystem(0)
    
    # For testing with video file
    # system = TrafficViolationSystem("traffic.mp4")
    
    # For RTSP stream
    # system = TrafficViolationSystem("rtsp://username:password@ip_address:port")
    
    system = TrafficViolationSystem()
    system.start()
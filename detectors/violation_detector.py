import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from config.settings import DETECTION_CONFIG

class ViolationDetector:
    def __init__(self):
        self.min_stop_time = DETECTION_CONFIG['min_stop_time']
        self.confidence_threshold = DETECTION_CONFIG['confidence_threshold']
        self.yellow_lower, self.yellow_upper = DETECTION_CONFIG['yellow_box_color_range']
        self.zebra_area_threshold = DETECTION_CONFIG['zebra_crossing_contour_area']
        
        # Track vehicles in restricted zones
        self.vehicles_in_yellow_box = defaultdict(dict)
        self.vehicles_in_zebra_crossing = defaultdict(dict)
        
        # Load YOLO model for vehicle detection
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Vehicle class IDs in COCO dataset (car, truck, bus, etc.)
        self.vehicle_class_ids = [2, 3, 5, 7]
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLO model"""
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Process detections
        vehicles = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold and class_id in self.vehicle_class_ids:
                    # Vehicle detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    vehicles.append({
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y)
                    })
        
        return vehicles
    
    def detect_yellow_boxes(self, frame):
        """Detect yellow box junctions using color thresholding"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.yellow_lower), np.array(self.yellow_upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(cnt)
                yellow_boxes.append((x, y, w, h))
        
        return yellow_boxes
    
    def detect_zebra_crossings(self, frame):
        """Detect zebra crossings using pattern recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        
        if lines is not None:
            # Find parallel lines with similar spacing (zebra pattern)
            # This is simplified - a real implementation would be more complex
            zebra_contours = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Simple approach: look for clusters of parallel lines
                # In a real system, we'd use more sophisticated pattern recognition
                if abs(y2 - y1) < 10:  # Horizontal lines
                    zebra_contours.append(line[0])
            
            if len(zebra_contours) > 5:  # Minimum lines to consider as zebra
                # Get bounding rect of all lines
                all_points = np.array(zebra_contours).reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(all_points)
                if w * h > self.zebra_area_threshold:
                    return [(x, y, w, h)]
        
        return []
    
    def check_violations(self, frame, vehicles, yellow_boxes, zebra_crossings):
        """Check for vehicles violating traffic rules"""
        violations = []
        current_time = datetime.now()
        
        # Check yellow box violations
        for box in yellow_boxes:
            bx, by, bw, bh = box
            box_rect = (bx, by, bx + bw, by + bh)
            
            for vehicle in vehicles:
                vx, vy, vw, vh = vehicle['bbox']
                vehicle_rect = (vx, vy, vx + vw, vy + vh)
                
                if self._rect_overlap(box_rect, vehicle_rect):
                    vehicle_id = f"{vx}_{vy}"
                    
                    if vehicle_id not in self.vehicles_in_yellow_box:
                        # New vehicle in yellow box
                        self.vehicles_in_yellow_box[vehicle_id] = {
                            'entry_time': current_time,
                            'last_seen': current_time,
                            'bbox': vehicle['bbox']
                        }
                    else:
                        # Update last seen time
                        self.vehicles_in_yellow_box[vehicle_id]['last_seen'] = current_time
                        duration = (current_time - self.vehicles_in_yellow_box[vehicle_id]['entry_time']).total_seconds()
                        
                        if duration >= self.min_stop_time:
                            # Violation detected
                            violations.append({
                                'type': 'yellow_box',
                                'vehicle': vehicle,
                                'duration': duration,
                                'location': box,
                                'vehicle_id': vehicle_id
                            })
        
        # Check zebra crossing violations (similar logic)
        for zebra in zebra_crossings:
            zx, zy, zw, zh = zebra
            zebra_rect = (zx, zy, zx + zw, zy + zh)
            
            for vehicle in vehicles:
                vx, vy, vw, vh = vehicle['bbox']
                vehicle_rect = (vx, vy, vx + vw, vy + vh)
                
                if self._rect_overlap(zebra_rect, vehicle_rect):
                    vehicle_id = f"{vx}_{vy}"
                    
                    if vehicle_id not in self.vehicles_in_zebra_crossing:
                        self.vehicles_in_zebra_crossing[vehicle_id] = {
                            'entry_time': current_time,
                            'last_seen': current_time,
                            'bbox': vehicle['bbox']
                        }
                    else:
                        self.vehicles_in_zebra_crossing[vehicle_id]['last_seen'] = current_time
                        duration = (current_time - self.vehicles_in_zebra_crossing[vehicle_id]['entry_time']).total_seconds()
                        
                        if duration >= self.min_stop_time:
                            violations.append({
                                'type': 'zebra_crossing',
                                'vehicle': vehicle,
                                'duration': duration,
                                'location': zebra,
                                'vehicle_id': vehicle_id
                            })
        
        # Clean up old entries
        self._cleanup_old_entries(current_time)
        
        return violations
    
    def _rect_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        # Check if one rectangle is to the left of the other
        if x2 < x3 or x4 < x1:
            return False
        
        # Check if one rectangle is above the other
        if y2 < y3 or y4 < y1:
            return False
        
        return True
    
    def _cleanup_old_entries(self, current_time):
        """Remove vehicles that haven't been seen for a while"""
        timeout = timedelta(seconds=self.min_stop_time * 2)
        
        # Clean yellow box entries
        to_remove = []
        for vehicle_id, data in self.vehicles_in_yellow_box.items():
            if (current_time - data['last_seen']) > timeout:
                to_remove.append(vehicle_id)
        
        for vehicle_id in to_remove:
            del self.vehicles_in_yellow_box[vehicle_id]
        
        # Clean zebra crossing entries
        to_remove = []
        for vehicle_id, data in self.vehicles_in_zebra_crossing.items():
            if (current_time - data['last_seen']) > timeout:
                to_remove.append(vehicle_id)
        
        for vehicle_id in to_remove:
            del self.vehicles_in_zebra_crossing[vehicle_id]
import cv2
import numpy as np

def draw_violation_info(frame, violation_info, license_plate):
    """Draw violation information on the frame"""
    x, y, w, h = violation_info['vehicle']['bbox']
    violation_type = violation_info['type']
    duration = violation_info['duration']
    
    # Draw bounding box
    color = (0, 0, 255)  # Red
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw violation info
    info_text = f"{violation_type} violation: {duration:.1f}s"
    cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw license plate if available
    if license_plate:
        cv2.putText(frame, f"Plate: {license_plate}", (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def draw_detection_zones(frame, yellow_boxes, zebra_crossings):
    """Draw detection zones on the frame"""
    # Draw yellow boxes
    for box in yellow_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Yellow Box", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw zebra crossings
    for zebra in zebra_crossings:
        x, y, w, h = zebra
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "Zebra Crossing", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
import cv2
import easyocr
import numpy as np
from config.settings import LP_CONFIG

class LicensePlateRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.min_width = LP_CONFIG['min_width']
        self.min_height = LP_CONFIG['min_height']
        self.max_width = LP_CONFIG['max_width']
        self.max_height = LP_CONFIG['max_height']
        self.contrast = LP_CONFIG['contrast_enhance']
        self.brightness = LP_CONFIG['brightness_enhance']
    
    def preprocess_plate(self, plate_img):
        """Enhance license plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast and brightness
        enhanced = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=self.brightness)
        
        # Apply thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def detect_license_plate(self, vehicle_img):
        """Detect and recognize license plate from vehicle image"""
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(blurred, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and get top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contour = None
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # Look for rectangular contours
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if contour dimensions match license plate proportions
                if (self.min_width < w < self.max_width and 
                    self.min_height < h < self.max_height and 
                    w > h and  # License plates are typically wider than tall
                    2 < w/h < 5):
                    
                    plate_contour = contour
                    break
        
        if plate_contour is not None:
            # Extract license plate region
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate_region = vehicle_img[y:y+h, x:x+w]
            
            # Preprocess for OCR
            processed_plate = self.preprocess_plate(plate_region)
            
            # Perform OCR
            results = self.reader.readtext(processed_plate, detail=0, paragraph=True)
            
            if results:
                # Join all detected text and clean up
                license_plate = ' '.join(results).strip().upper()
                license_plate = ''.join(c for c in license_plate if c.isalnum() or c.isspace())
                return license_plate, (x, y, w, h)
        
        return None, None
    
    def recognize_from_frame(self, frame, vehicle_bbox):
        """Recognize license plate from frame given vehicle bounding box"""
        x, y, w, h = vehicle_bbox
        vehicle_img = frame[y:y+h, x:x+w]
        return self.detect_license_plate(vehicle_img)
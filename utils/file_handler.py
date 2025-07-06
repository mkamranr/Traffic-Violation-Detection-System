import os
import cv2
from datetime import datetime
from config.settings import STORAGE_CONFIG

class FileHandler:
    def __init__(self):
        self.images_dir = STORAGE_CONFIG['images_dir']
        self.videos_dir = STORAGE_CONFIG['videos_dir']
        self.max_storage_mb = STORAGE_CONFIG['max_storage_mb']
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
    
    def save_violation_image(self, frame, violation_id):
        """Save violation image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_id}_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        cv2.imwrite(filepath, frame)
        return filepath
    
    def save_violation_video(self, frames, violation_id, fps=20):
        """Save violation video clip to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_id}_{timestamp}.avi"
        filepath = os.path.join(self.videos_dir, filename)
        
        if not frames:
            return None
        
        # Get frame dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return filepath
    
    def check_storage(self):
        """Check if storage is within limits"""
        total_size = 0
        
        for dirpath, _, filenames in os.walk(self.images_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        for dirpath, _, filenames in os.walk(self.videos_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > self.max_storage_mb:
            # Delete oldest files until under limit
            self._cleanup_storage()
    
    def _cleanup_storage(self):
        """Delete oldest files to free up space"""
        # Get all files with their creation times
        all_files = []
        
        for dirpath, _, filenames in os.walk(self.images_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                all_files.append((fp, os.path.getctime(fp)))
        
        for dirpath, _, filenames in os.walk(self.videos_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                all_files.append((fp, os.path.getctime(fp)))
        
        # Sort by creation time (oldest first)
        all_files.sort(key=lambda x: x[1])
        
        # Delete files until under limit
        total_size = sum(os.path.getsize(f[0]) for f in all_files)
        max_size_bytes = self.max_storage_mb * 1024 * 1024
        
        while total_size > max_size_bytes and all_files:
            oldest_file = all_files.pop(0)
            total_size -= os.path.getsize(oldest_file[0])
            os.remove(oldest_file[0])
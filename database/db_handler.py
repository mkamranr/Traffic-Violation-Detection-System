from pymongo import MongoClient
from datetime import datetime
from config.settings import MONGO_CONFIG
import uuid

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(
            host=MONGO_CONFIG['host'],
            port=MONGO_CONFIG['port']
        )
        self.db = self.client[MONGO_CONFIG['db_name']]
        self.collection = self.db[MONGO_CONFIG['collection']]
        
    def create_violation_record(self, violation_data):
        """Create a new violation record in MongoDB"""
        violation_id = str(uuid.uuid4())
        record = {
            '_id': violation_id,
            'timestamp': datetime.now(),
            'license_plate': violation_data.get('license_plate', 'UNKNOWN'),
            'violation_type': violation_data.get('violation_type'),
            'location': violation_data.get('location'),
            'duration': violation_data.get('duration'),
            'image_path': violation_data.get('image_path'),
            'video_path': violation_data.get('video_path'),
            'status': 'pending'
        }
        
        self.collection.insert_one(record)
        return violation_id
    
    def get_violation_by_id(self, violation_id):
        """Retrieve a violation record by ID"""
        return self.collection.find_one({'_id': violation_id})
    
    def update_violation_status(self, violation_id, status):
        """Update the status of a violation"""
        self.collection.update_one(
            {'_id': violation_id},
            {'$set': {'status': status}}
        )
    
    def close_connection(self):
        """Close MongoDB connection"""
        self.client.close()
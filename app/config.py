import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('/home/manogyaguragai/Desktop/Projects/talynx_backend/.env')

MONGODB_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB_NAME", "KandidexDB")
JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))

def get_db():
    client = MongoClient(MONGODB_URL)
    return client[DB_NAME]

# Collection helpers
def get_user_collection():
    return get_db().users

def get_job_details_collection():
    return get_db().job_details

def get_resumes_collection():
    return get_db().resumes

def get_batches_collection():
    return get_db().batch

def get_screening_runs_collection():
    return get_db().screening_runs

def get_activity_logs_collection():
    return get_db().activity_logs

def get_settings_collection():
    return get_db().settings
# Activity logging
def log_activity(user_id: str, activity_type: str, details: str, ref_id: str = None):
    activity = {
        "user_id": user_id,
        "type": activity_type,
        "ref_id": ref_id,
        "timestamp": datetime.now(),
        "details": details
    }
    get_activity_logs_collection().insert_one(activity)
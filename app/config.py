import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv('/home/manogyaguragai/Desktop/Projects/talynx_backend/.env')

MONGODB_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB_NAME", "KandidexDB")
JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))

print(MONGODB_URL)



def get_db():
    client = MongoClient(MONGODB_URL)
    return client[DB_NAME]

def get_user_collection():
    db = get_db()
    return db.users

def get_ranking_collection():
    db = get_db()
    return db.rankings
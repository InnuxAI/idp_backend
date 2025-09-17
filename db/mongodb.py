import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = None
        self.database = None

# MongoDB connection
mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
        mongodb.database = mongodb.client.idp
        
        # Test connection
        await mongodb.client.admin.command('ping')
        print("✅ Successfully connected to MongoDB")
        
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        raise e
    except Exception as e:
        print(f"❌ Unexpected error connecting to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        print("✅ MongoDB connection closed")

def get_database():
    """Get database instance"""
    return mongodb.database

"""
database.py - MongoDB connection and collection accessors
Runs against local MongoDB instance (mongodb://localhost:27017)
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = "vibration_optimizer"

_client: MongoClient = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
    return _client


def get_db():
    return get_client()[DB_NAME]


# ── Collection helpers ────────────────────────────────────────────────────────

def users_col():
    return get_db()["users"]


def runs_col():
    return get_db()["optimization_runs"]


def saved_col():
    return get_db()["saved_configs"]


# ── Index creation on startup ─────────────────────────────────────────────────

def ensure_indexes():
    users_col().create_index([("email", ASCENDING)], unique=True)
    runs_col().create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    saved_col().create_index([("user_id", ASCENDING)])


def check_connection() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except ConnectionFailure:
        return False

#!/usr/bin/env python3
"""
Script to find existing user in the database.
Run this to see what user exists in the system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.user import User
from app.core.config import settings

def find_users():
    """Find all users in the database"""
    try:
        # Create engine
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create session
        db = SessionLocal()
        
        try:
            # Query all users
            users = db.query(User).all()
            
            if not users:
                print("No users found in the database.")
                return
            
            print(f"Found {len(users)} user(s):")
            print("=" * 50)
            
            for user in users:
                print(f"ID: {user.id}")
                print(f"Username: {user.username}")
                print(f"Email: {user.email}")
                print(f"Active: {user.is_active}")
                print(f"Created: {user.created_at}")
                if user.last_login_at:
                    print(f"Last login: {user.last_login_at}")
                else:
                    print("Last login: Never")
                print("-" * 30)
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error querying database: {e}")
        print("Make sure:")
        print("1. Database is running (docker-compose up -d db)")
        print("2. Environment variables are set correctly")
        print("3. Dependencies are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    find_users()
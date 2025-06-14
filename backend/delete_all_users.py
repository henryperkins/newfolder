#!/usr/bin/env python3
"""
Script to delete all users from the database.
WARNING: This will remove all user data and allow fresh registration.
Use this only if you want to completely reset the user system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import User
from app.core.config import settings

def delete_all_users():
    """Delete all users from the database"""
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
            for user in users:
                print(f"- {user.username} ({user.email})")
            
            # Confirm deletion
            confirm = input(f"\nAre you sure you want to delete ALL {len(users)} user(s)? This cannot be undone! (yes/no): ")
            
            if confirm.lower() != 'yes':
                print("Deletion cancelled.")
                return
            
            # Double confirmation
            confirm2 = input("Type 'DELETE ALL USERS' to confirm: ")
            if confirm2 != 'DELETE ALL USERS':
                print("Deletion cancelled.")
                return
            
            # Delete all users
            deleted_count = db.query(User).delete()
            db.commit()
            
            print(f"\nSuccessfully deleted {deleted_count} user(s).")
            print("Registration is now open again.")
            print("You can now create a new user account through the web interface.")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. Database is running (docker-compose up -d db)")
        print("2. Environment variables are set correctly")
        print("3. Dependencies are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    print("USER DELETION UTILITY")
    print("=" * 40)
    print("WARNING: This will delete ALL users!")
    print("=" * 40)
    delete_all_users()
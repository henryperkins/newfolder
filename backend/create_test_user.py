#!/usr/bin/env python3
"""
Script to create a test user directly in the database using SQL.
This bypasses the ORM relationship issues.
"""

import sys
import os
import hashlib
import uuid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from app.core.config import settings

def hash_password(password: str) -> str:
    """Simple password hashing using bcrypt-like approach"""
    import bcrypt
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_test_user():
    """Create a test user directly via SQL"""
    try:
        # Create engine
        engine = create_engine(settings.database_url)
        
        # User details
        user_id = str(uuid.uuid4())
        username = "admin"
        email = "admin@example.com"
        password = "admin123"
        hashed_password = hash_password(password)
        
        with engine.connect() as connection:
            # Insert user
            insert_sql = text("""
                INSERT INTO users (id, username, email, hashed_password, is_active, created_at, updated_at)
                VALUES (:id, :username, :email, :hashed_password, :is_active, now(), now())
            """)
            
            connection.execute(insert_sql, {
                'id': user_id,
                'username': username,
                'email': email,
                'hashed_password': hashed_password,
                'is_active': True
            })
            
            connection.commit()
            
            print("✅ Test user created successfully!")
            print(f"Username: {username}")
            print(f"Email: {email}")
            print(f"Password: {password}")
            print(f"User ID: {user_id}")
            print()
            print("You can now log in with these credentials:")
            print(f"  Email: {email}")
            print(f"  Password: {password}")
                
    except Exception as e:
        print(f"Error creating user: {e}")
        print("Make sure:")
        print("1. Database is running (docker-compose up -d db)")
        print("2. User table exists (run create_user_table.py first)")

if __name__ == "__main__":
    create_test_user()
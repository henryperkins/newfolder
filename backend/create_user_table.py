#!/usr/bin/env python3
"""
Script to create just the users table and query for existing users.
This bypasses the complex chat system relationships.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from app.core.config import settings

def create_user_table_and_query():
    """Create users table if it doesn't exist and query for users"""
    try:
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as connection:
            # Create users table
            create_table_sql = text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    username VARCHAR(30) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT true NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                    last_login_at TIMESTAMP WITH TIME ZONE,
                    CONSTRAINT username_min_length CHECK (char_length(username) >= 3),
                    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$')
                );
                
                CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
                CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);
            """)
            
            connection.execute(create_table_sql)
            connection.commit()
            
            # Query for existing users
            result = connection.execute(text("SELECT id, username, email, is_active, created_at, last_login_at FROM users"))
            users = result.fetchall()
            
            if not users:
                print("No users found in the database.")
                print("Registration is open - you can create a new user account.")
                return
            
            print(f"Found {len(users)} user(s):")
            print("=" * 60)
            
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
                print("-" * 40)
                
            print("These are the existing user credentials in the system.")
            print("If you don't know the password, you can:")
            print("1. Use the 'Forgot Password' feature in the web interface")
            print("2. Run the reset_user.py script to reset the password")
            print("3. Run the delete_all_users.py script to start fresh (WARNING: deletes all data)")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. Database is running (docker-compose up -d db)")
        print("2. PostgreSQL is accessible")

if __name__ == "__main__":
    create_user_table_and_query()
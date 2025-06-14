#!/usr/bin/env python3
"""
Script to reset user password or create a new user if the database is empty.
This is a utility script to help recover from locked out situations.
"""

import sys
import os
import getpass
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import User
from app.core.config import settings
from app.services.security import SecurityService

def reset_or_create_user():
    """Reset existing user password or create new user if none exists"""
    try:
        # Create engine
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create session
        db = SessionLocal()
        security_service = SecurityService()
        
        try:
            # Check if any users exist
            users = db.query(User).all()
            
            if not users:
                print("No users found. Creating a new user...")
                
                # Get user details
                print("\nEnter details for the new user:")
                username = input("Username: ").strip()
                if not username:
                    print("Username cannot be empty!")
                    return
                    
                email = input("Email: ").strip()
                if not email:
                    print("Email cannot be empty!")
                    return
                    
                password = getpass.getpass("Password: ")
                if not password:
                    print("Password cannot be empty!")
                    return
                
                # Create user
                hashed_password = security_service.hash_password(password)
                new_user = User(
                    username=username,
                    email=email,
                    hashed_password=hashed_password,
                    is_active=True
                )
                
                db.add(new_user)
                db.commit()
                
                print(f"\nUser '{username}' created successfully!")
                print(f"Email: {email}")
                print("You can now log in with these credentials.")
                
            else:
                print(f"Found {len(users)} user(s):")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user.username} ({user.email})")
                
                # Select user to reset
                if len(users) == 1:
                    selected_user = users[0]
                    print(f"\nSelected user: {selected_user.username}")
                else:
                    while True:
                        try:
                            choice = int(input(f"\nSelect user to reset password (1-{len(users)}): "))
                            if 1 <= choice <= len(users):
                                selected_user = users[choice - 1]
                                break
                            else:
                                print("Invalid choice!")
                        except ValueError:
                            print("Please enter a number!")
                
                # Get new password
                new_password = getpass.getpass(f"Enter new password for {selected_user.username}: ")
                if not new_password:
                    print("Password cannot be empty!")
                    return
                
                # Update password
                selected_user.hashed_password = security_service.hash_password(new_password)
                selected_user.is_active = True  # Ensure account is active
                
                db.commit()
                
                print(f"\nPassword updated successfully for user '{selected_user.username}'!")
                print(f"Email: {selected_user.email}")
                print("You can now log in with the new password.")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. Database is running (docker-compose up -d db)")
        print("2. Environment variables are set correctly")
        print("3. Dependencies are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    print("User Management Utility")
    print("=" * 30)
    reset_or_create_user()
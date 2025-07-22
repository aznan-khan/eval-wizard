"""
Authentication API routes (simplified for demo)
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.post("/login")
async def login(credentials: Dict[str, str]):
    """Simple login endpoint for demo"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Demo authentication (replace with real auth in production)
    if username == "admin" and password == "admin":
        return {
            "access_token": "demo_token_123",
            "token_type": "bearer",
            "expires_in": 3600
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/profile")
async def get_profile():
    """Get user profile"""
    return {
        "username": "admin",
        "role": "analyst",
        "permissions": ["read", "write", "analyze"]
    }
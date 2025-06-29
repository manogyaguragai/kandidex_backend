from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from config import get_user_collection, get_settings_collection
from utils.security import (
    verify_password, 
    create_access_token, 
    get_password_hash
)
from models.auth import UserCreate, User, Token
import uuid
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(tags=["authentication"])

class PasswordReset(BaseModel):
    email: str
    old_password: str
    new_password: str
    
@router.post("/register", response_model=User)
async def register(user: UserCreate):
    users_collection = get_user_collection()
    
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_dict = user.dict()
    # user_dict["id"] = str(uuid.uuid4())
    user_dict["hashed_password"] = get_password_hash(user.password)
    user_dict["created_at"] = datetime.now()
    user_dict["disabled"] = False
    
    # Remove plain password before storing
    del user_dict["password"]
    
    result = users_collection.insert_one(user_dict)
    settings_collection = get_settings_collection()
    settings_collection.insert_one(
        {"user_id": str(result.inserted_id),
        "created_at": datetime.now(),
        "number_of_questions_to_generate": 10,
        "phase1_ranking_number": 20,
        "phase2_ranking_number": 10,
        "updated_at": datetime.now()}
        )
    
    print("USER Created")
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    return user_dict

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users_collection = get_user_collection()
    
    if users_collection is not None:
        print("USER Collection Found")
    
    print("USER Login")
    user = users_collection.find_one({"email": form_data.username})

    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user["disabled"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["email"]})
    user_id = str(user["_id"])
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id}

@router.post("/reset-password", response_model=dict)
async def reset_password(reset_data: PasswordReset):
    users_collection = get_user_collection()
    
    # Fetch user by email
    user = users_collection.find_one({"email": reset_data.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found with this email address"
        )
    
    # Check account status
    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is disabled. Contact support for assistance."
        )
    
    # Verify old password
    if not verify_password(reset_data.old_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The password you entered is incorrect. Please try again."
        )
    
    # Prevent password reuse
    if verify_password(reset_data.new_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from your current password"
        )
    
    # Validate new password strength (optional)
    if len(reset_data.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Update password
    new_hashed_password = get_password_hash(reset_data.new_password)
    result = users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "hashed_password": new_hashed_password,
            "updated_at": datetime.now()
        }}
    )
    
    if result.modified_count == 1:
        return {"message": "Password updated successfully!"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password update failed. Please try again later."
        )
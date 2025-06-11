from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from config import get_user_collection
from utils.security import (
    verify_password, 
    create_access_token, 
    get_password_hash
)
from models.auth import UserCreate, User, Token
import uuid
from datetime import datetime

router = APIRouter(tags=["authentication"])

@router.post("/register", response_model=User)
async def register(user: UserCreate):
    print("USER Creation")
    users_collection = get_user_collection()
    
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_dict = user.dict()
    # user_dict["id"] = str(uuid.uuid4())
    user_dict["hashed_password"] = get_password_hash(user.password)
    user_dict["created_at"] = datetime.utcnow()
    user_dict["disabled"] = False
    
    # Remove plain password before storing
    del user_dict["password"]
    
    result = users_collection.insert_one(user_dict)
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
    
    access_token = create_access_token(data={"sub": user["email"]})
    user_id = str(user["_id"])
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id}
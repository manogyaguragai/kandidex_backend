from fastapi import APIRouter, HTTPException, Form
from datetime import datetime
from config import get_settings_collection, get_user_collection, log_activity
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/settings", tags=["settings"])

class UserSettings(BaseModel):
    phase1_ranking_number: Optional[int]
    phase2_ranking_number: Optional[int]
    number_of_questions_to_generate: Optional[int]

@router.post("/")
async def update_ranking_settings(
    user_id: str = Form(...),
    phase1_ranking_number: Optional[int] = Form(...),
    phase2_ranking_number: Optional[int] = Form(...),
    number_of_questions_to_generate: Optional[int] = Form(...),
):
    """
    Update or create ranking settings for a user
    - Stores phase1_ranking_number and phase2_ranking_number in MongoDB
    - Uses upsert to create if not exists, update if exists
    - Validates input numbers
    """
    
    # Validate input numbers
    if phase1_ranking_number <= 0 or phase2_ranking_number <= 0:
        raise HTTPException(
            status_code=400,
            detail="Ranking numbers must be positive integers"
        )
    
    if number_of_questions_to_generate is not None and number_of_questions_to_generate < 1:
        raise HTTPException(
            status_code=400,
            detail="Number of questions to generate must be a positive integer"
        )
    
    if phase1_ranking_number < phase2_ranking_number:
        raise HTTPException(
            status_code=400,
            detail="Phase 1 ranking number must be greater than or equal to Phase 2"
        )
    
    user = get_user_collection().find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    # Prepare update document
    update_data = {
        "phase1_ranking_number": phase1_ranking_number,
        "phase2_ranking_number": phase2_ranking_number,
        "number_of_questions_to_generate": number_of_questions_to_generate,
        "updated_at": datetime.now()
    }
    
    # MongoDB upsert operation
    try:
        result = get_settings_collection().update_one(
            {"user_id": user_id},
            {"$set": update_data, 
             "$setOnInsert": {"created_at": datetime.now()}},
            upsert=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    
    # Prepare log message based on operation type
    action = "created" if result.upserted_id else "updated"
    log_message = (
        f"Ranking settings {action}: "
        f"Phase1={phase1_ranking_number}, Phase2={phase2_ranking_number}"
    )
    
    # Log activity
    log_activity(user_id, "ranking_settings_update", log_message)
    
    return {
        "status": "success",
        "message": log_message,
        "user_id": user_id,
        "phase1_ranking_number": phase1_ranking_number,
        "phase2_ranking_number": phase2_ranking_number,
        "number_of_questions_to_generate": number_of_questions_to_generate
    }
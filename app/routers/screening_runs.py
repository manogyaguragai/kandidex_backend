from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
from config import get_screening_runs_collection
from bson import ObjectId

router = APIRouter(prefix="/screening_runs", tags=["screening_runs"])

# Define response models
class SkillAssessment(BaseModel):
    exact_matches: List[str]
    transferable_skills: List[str]
    non_technical_skills: List[str]

class GeneratedQuestion(BaseModel):
    question: str
    skill_type: str
    difficulty: str

class ScreeningCandidate(BaseModel):
    resume_id: str
    candidate_name: str
    file_name: str
    ai_fit_score: float
    skill_similarity: float
    candidate_summary: str
    skill_assessment: SkillAssessment
    experience_highlights: str
    education_highlights: str
    gaps: List[str]
    ai_justification: str
    resume_content_preview: str
    questions_generated: bool
    generated_questions: List[GeneratedQuestion]

class ScreeningRunResponse(BaseModel):
    id: str
    job_details_id: str
    batch_id: str
    run_start_time: datetime
    run_end_time: datetime
    time_taken: float  # in seconds
    created_at: datetime
    candidates: List[ScreeningCandidate]

@router.get("/", response_model=List[ScreeningRunResponse])
async def get_screening_runs(
    user_id: str = Query(..., description="User ID to fetch screening runs for"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date in ISO format (YYYY-MM-DD)")
):
    # Build query filter
    query = {"user_id": user_id}
    
    # Add date filters if provided
    date_filter = {}
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            date_filter["$gte"] = start_dt
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
            date_filter["$lt"] = end_dt
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    if date_filter:
        query["created_at"] = date_filter
    
    # Fetch screening runs from database
    collection = get_screening_runs_collection()
    runs = list(collection.find(query).sort("created_at", -1))
    
    if not runs:
        raise HTTPException(status_code=404, detail="No screening runs found for this user and filters")
    
    # Process and format the response
    response = []
    for run in runs:
        # Calculate time taken for screening
        time_taken = (run["run_end_time"] - run["run_start_time"]).total_seconds()
        
        # Process candidates
        candidates = []
        for candidate in run.get("candidates", []):
            # Handle skill_assessment structure
            skill_assessment = candidate.get("skill_assessment", {})
            if not isinstance(skill_assessment, dict):
                skill_assessment = {}
                
            candidates.append(ScreeningCandidate(
                resume_id=candidate["resume_id"],
                candidate_name=candidate["candidate_name"],
                file_name=candidate["file_name"],
                ai_fit_score=candidate["ai_fit_score"],
                skill_similarity=candidate["skill_similarity"],
                candidate_summary=candidate["candidate_summary"],
                skill_assessment=SkillAssessment(
                    exact_matches=skill_assessment.get("exact_matches", []),
                    transferable_skills=skill_assessment.get("transferable_skills", []),
                    non_technical_skills=skill_assessment.get("non_technical_skills", [])
                ),
                experience_highlights=candidate["experience_highlights"],
                education_highlights=candidate["education_highlights"],
                gaps=candidate["gaps"],
                ai_justification=candidate["ai_justification"],
                resume_content_preview=candidate["resume_content_preview"],
                questions_generated=candidate["questions_generated"],
                generated_questions=[
                    GeneratedQuestion(**q) for q in candidate.get("generated_questions", [])
                ]
            ))
        
        response.append(ScreeningRunResponse(
            id=str(run["_id"]),
            job_details_id=run["job_details_id"],
            batch_id=run["batch_id"],
            run_start_time=run["run_start_time"],
            run_end_time=run["run_end_time"],
            time_taken=time_taken,
            created_at=run["created_at"],
            candidates=candidates
        ))
    
    return response
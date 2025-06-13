from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional
from config import get_screening_runs_collection, get_job_details_collection
from bson import ObjectId
import math

router = APIRouter(prefix="/screening_runs", tags=["screening_runs"])

# Define response models
class SkillAssessment(BaseModel):
    exact_matches: List[str] = Field(..., description="List of exact matching skills")
    transferable_skills: List[str] = Field(..., description="List of transferable skills")
    non_technical_skills: List[str] = Field(..., description="List of non-technical skills")

class GeneratedQuestion(BaseModel):
    question: str = Field(..., description="Generated interview question")
    skill_type: str = Field(..., description="Type of skill the question assesses")
    difficulty: str = Field(..., description="Difficulty level of the question")

class ScreeningCandidate(BaseModel):
    resume_id: str = Field(..., description="Unique identifier for the resume")
    candidate_name: str = Field(..., description="Name of the candidate")
    file_name: str = Field(..., description="Original filename of the resume")
    ai_fit_score: float = Field(..., description="AI-generated fit score for the candidate")
    skill_similarity: float = Field(..., description="Similarity score between candidate skills and job requirements")
    candidate_summary: str = Field(..., description="Brief summary of the candidate")
    skill_assessment: SkillAssessment = Field(..., description="Detailed skill assessment")
    experience_highlights: str = Field(..., description="Key experience highlights")
    education_highlights: str = Field(..., description="Key education highlights")
    gaps: List[str] = Field(..., description="List of identified skill/experience gaps")
    ai_justification: str = Field(..., description="AI justification for the fit score")
    resume_content_preview: str = Field(..., description="Preview of resume content")
    questions_generated: bool = Field(..., description="Flag indicating if questions were generated")
    generated_questions: List[GeneratedQuestion] = Field(..., description="List of generated interview questions")

class ScreeningRunResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the screening run")
    job_details_id: str = Field(..., description="Identifier for the job details")
    job_role: Optional[str] = Field(None, description="Job role from job details")
    job_description: Optional[str] = Field(None, description="Job description from job details")
    batch_id: str = Field(..., description="Batch identifier for the run")
    run_start_time: datetime = Field(..., description="Start time of the screening run")
    run_end_time: datetime = Field(..., description="End time of the screening run")
    time_taken: float = Field(..., description="Duration of the run in seconds")
    created_at: datetime = Field(..., description="Creation timestamp of the run")
    candidates: List[ScreeningCandidate] = Field(..., description="List of screened candidates")

class PaginatedScreeningRunResponse(BaseModel):
    total: int = Field(..., description="Total number of screening runs")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    results: List[ScreeningRunResponse] = Field(..., description="List of screening runs")

@router.get("/", response_model=PaginatedScreeningRunResponse)
async def get_screening_runs(
    user_id: str = Query(..., description="User ID to fetch screening runs for"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date in ISO format (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page")
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
    
    # Fetch screening runs collection
    screening_runs_collection = get_screening_runs_collection()
    job_details_collection = get_job_details_collection()
    
    # Count total documents matching the query
    total = screening_runs_collection.count_documents(query)
    
    # Calculate pagination values
    total_pages = math.ceil(total / limit) if total > 0 else 1
    skip = (page - 1) * limit
    
    # Fetch paginated screening runs
    runs = list(
        screening_runs_collection.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    
    if not runs:
        return PaginatedScreeningRunResponse(
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            results=[]
        )
    
    # Process and format the response
    results = []
    for run in runs:
        # Fetch job details
        job_role = None
        job_description = None
        if run.get("job_details_id"):
            job_details = job_details_collection.find_one(
                {"_id": ObjectId(run["job_details_id"])}
            )
            if job_details:
                job_role = job_details.get("job_role")
                job_description = job_details.get("job_description")
        
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
        
        results.append(ScreeningRunResponse(
            id=str(run["_id"]),
            job_details_id=run["job_details_id"],
            job_role=job_role,
            job_description=job_description,
            batch_id=run["batch_id"],
            run_start_time=run["run_start_time"],
            run_end_time=run["run_end_time"],
            time_taken=time_taken,
            created_at=run["created_at"],
            candidates=candidates
        ))
    
    return PaginatedScreeningRunResponse(
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        results=results
    )
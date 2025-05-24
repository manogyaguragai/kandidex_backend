from fastapi import APIRouter
from services.question_service import generate_questions

router = APIRouter(prefix="/generate_questions", tags=["Questions"])

@router.get("/{resume_id}")
async def get_questions(resume_id: str):
    return await generate_questions(resume_id)

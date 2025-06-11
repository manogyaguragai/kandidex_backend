from fastapi import APIRouter, HTTPException, Form, Query
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal
from openai import OpenAI
import os
import json
import re
from config import (
    get_screening_runs_collection,
    get_resumes_collection,
    get_job_details_collection,
    log_activity
)
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/questions", tags=["generate_questions"])

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str
    skill_type: Literal["soft skill", "hard skill"]
    difficulty: Literal["entry level", "mid level", "senior"]

class QuestionGroup(BaseModel):
    candidate_name: str
    questions: List[Question]

def extract_json_from_response(raw_output: str) -> str:
    if '```json' in raw_output:
        json_match = re.search(r'```json(.*?)```', raw_output, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

    if '```' in raw_output:
        code_match = re.search(r'```(.*?)```', raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

    obj_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()

    return raw_output.strip()

def sanitize_json(raw_json: str) -> str:
    sanitized = re.sub(r',\s*([}\]])', r'\1', raw_json)
    sanitized = re.sub(r'([,{])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', sanitized)
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    return sanitized

def update_screening_run_with_questions(run_id: str, resume_id: str, questions: List[dict]):
    # Find and update the specific candidate in the screening run
    screening_runs = get_screening_runs_collection()
    
    # Find the run and candidate
    run = screening_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(404, "Screening run not found")
    
    # Find candidate in run
    candidate_index = None
    for idx, candidate in enumerate(run["candidates"]):
        if candidate["resume_id"] == resume_id:
            candidate_index = idx
            break
    
    if candidate_index is None:
        raise HTTPException(404, "Candidate not found in screening run")
    
    # Update the candidate document
    update_query = {
        "$set": {
            f"candidates.{candidate_index}.questions_generated": True,
            f"candidates.{candidate_index}.generated_questions": questions
        }
    }
    
    screening_runs.update_one({"_id": ObjectId(run_id)}, update_query)

@router.post("/")
async def generate_questions(
    user_id: str = Form(...),
    screening_run_id: str = Form(...),
    resume_id: str = Form(...),
    num_questions: int = Query(5, description="Number of questions to generate"),
    soft_skills_flag: bool = Query(False, description="Include soft skill questions?"),
    hard_skills_flag: bool = Query(True, description="Include hard skill questions?"),
    soft_skills_focus: Optional[str] = Query(None, description="Focus areas for soft skills (comma-separated)"),
    hard_skills_focus: Optional[str] = Query(None, description="Focus areas for hard skills (comma-separated)"),
    include_coding: bool = Query(False, description="Should hard skills include coding questions?")
):
    # Get resume content
    resume_doc = get_resumes_collection().find_one({"_id": ObjectId(resume_id)})
    if not resume_doc:
        raise HTTPException(404, "Resume not found")
    
    # Get screening run to find job details
    screening_run = get_screening_runs_collection().find_one(
        {"_id": ObjectId(screening_run_id)}
    )
    if not screening_run:
        raise HTTPException(404, "Screening run not found")
    
    # Get job description from job details
    job_details = get_job_details_collection().find_one(
        {"_id": ObjectId(screening_run["job_details_id"])}
    )
    if not job_details:
        raise HTTPException(404, "Job details not found")
    
    resume_content = resume_doc.get("content", "")
    candidate_name = resume_doc.get("candidate_name", "Candidate")
    job_description = job_details["job_description"]

    soft_skills_flag_str = "yes" if soft_skills_flag else "no"
    hard_skills_flag_str = "yes" if hard_skills_flag else "no"
    include_coding_str = "yes" if include_coding else "no"

    prompt = f"""
              You are an expert recruiter and technical interviewer. Your task is to generate a fixed number of tailored interview questions based on a candidate's resume and a job description.

              ---

              ### INPUTS:
              - Candidate Name: {candidate_name}
              - Resume Content: {resume_content[:2000]} [truncated]
              - Job Description: {job_description}
              - Number of Questions (optional, default = 5): {num_questions}
              - Include Soft Skills Questions? (optional, \"yes\" or \"no\", default = \"no\"): {soft_skills_flag_str}
              - Include Hard Skills Questions? (optional, \"yes\" or \"no\", default = \"yes\"): {hard_skills_flag_str}
              - Soft Skills Focus Areas (optional): {soft_skills_focus}
              (e.g., teamwork, leadership, communication, conflict resolution)
              - Hard Skills Emphasis (optional): {hard_skills_focus}
              (e.g., system design, cloud architecture, data engineering, ML)
              - Should Include Coding Questions? (optional, \"yes\" or \"no\", default = \"no\"): {include_coding_str}

              ---

              ### INSTRUCTIONS:

              1. Carefully analyze the resume and job description to determine the candidate's **experience level**:
              - Use role titles, years of experience, and project scope to infer whether the candidate is **entry level**, **mid level**, or **senior**.

              2. If `soft_skills_flag` is set to **\"yes\"**, generate at least one **soft skill** question:
              - Format it as a behavioral or scenario-based prompt.
              - Focus on real-world dynamics (e.g., collaboration, conflict resolution).
              - Use `soft_skills_focus` if provided to guide the theme.

              3. If `hard_skills_flag` is set to **\"yes\"**, generate technical questions relevant to:
              - The candidate’s experience (e.g., tools, frameworks, roles)
              - The job description
              - Use `hard_skills_focus` if provided to tailor content
              - If `include_coding` is \"yes\", include at least one coding/algorithmic problem

              4. You must return **exactly {num_questions}** questions (default = 5 if not provided).

              5. Each question must be returned in **JSON format**, with the following structure:

              ```json
              {{
                "candidate_name": "{candidate_name}",
                "questions": [
                  {{
                    "question": "Describe a time you had to lead a team through a high-pressure situation.",
                    "skill_type": "soft skill",
                    "difficulty": "senior"
                  }},
                  {{
                    "question": "Design a distributed caching strategy for a high-traffic e-commerce platform.",
                    "skill_type": "hard skill",
                    "difficulty": "senior"
                  }},
                  {{
                    "question": "Write a Python function that returns the first non-repeating character in a string.",
                    "skill_type": "hard skill",
                    "difficulty": "mid level"
                  }}
                ]
              }}
              ```

              Only use \"soft skill\" or \"hard skill\" for the "skill_type" field. Do not include commentary—just return the JSON.
              """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2500
    )

    raw_output = response.choices[0].message.content.strip()
    json_str = extract_json_from_response(raw_output)
    sanitized_json = sanitize_json(json_str)

    try:
        parsed = json.loads(sanitized_json)
        validated = QuestionGroup(**parsed)
        
        # Convert to dictionary for storage
        questions_dict = [q.dict() for q in validated.questions]
        
        # Update screening run
        update_screening_run_with_questions(
            run_id=screening_run_id,
            resume_id=resume_id,
            questions=questions_dict
        )
        
        # Log activity
        log_activity(
            user_id,
            "questions_generated",
            f"Generated {len(questions_dict)} questions for {candidate_name}",
            screening_run_id
        )
        
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        error_detail = {
            "error_type": type(e).__name__,
            "message": str(e),
            "raw_output_sample": raw_output[:500] + "..." if len(raw_output) > 500 else raw_output,
            "sanitized_json_sample": sanitized_json[:500] + "..." if len(sanitized_json) > 500 else sanitized_json
        }
        raise HTTPException(
            status_code=500,
            detail=f"JSON parsing failed: {json.dumps(error_detail, indent=2)}"
        )
from fastapi import APIRouter, HTTPException, Form, Query
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal
from openai import OpenAI
import os
import json
import re

router = APIRouter(prefix="/questions", tags=["generate_questions"])

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Keep the QuestionItem model for response validation
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

@router.post("/")
async def generate_questions(
    resume_content: str = Form(..., description="Raw text of the candidate's resume"),
    job_description: str = Form(..., description="Job description text"),
    num_questions: int = Query(5, description="Number of questions to generate"),
    soft_skills_flag: bool = Query(False, description="Include soft skill questions?"),
    hard_skills_flag: bool = Query(True, description="Include hard skill questions?"),
    soft_skills_focus: Optional[str] = Query(None, description="Focus areas for soft skills (comma-separated)"),
    hard_skills_focus: Optional[str] = Query(None, description="Focus areas for hard skills (comma-separated)"),
    include_coding: bool = Query(False, description="Should hard skills include coding questions?")
):
    soft_skills_flag_str = "yes" if soft_skills_flag else "no"
    hard_skills_flag_str = "yes" if hard_skills_flag else "no"
    include_coding_str = "yes" if include_coding else "no"

    prompt = f"""
              You are an expert recruiter and technical interviewer. Your task is to generate a fixed number of tailored interview questions based on a candidate's resume and a job description.

              ---

              ### INPUTS:
              - Resume Content: {resume_content}
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
                "candidate_name": "John Doe",
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
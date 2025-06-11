import re
import io
import zipfile
import fitz
import torch
import time
import json
from typing import List, Dict, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
from datetime import datetime
from config import (
    get_job_details_collection,
    get_resumes_collection,
    get_batches_collection,
    get_screening_runs_collection,
    log_activity
)
from bson import ObjectId

router = APIRouter(prefix="/rank", tags=["ranking"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
bi_encoder.max_seq_length = 512
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Extraction Helpers ---
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        return ""

def extract_contact_details(text: str) -> Dict[str, str]:
    """Extract email and phone via regex."""
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)
    return {"email": emails[0] if emails else "", "mobile_number": phones[0] if phones else ""}

def extract_name_with_llm(resume_text: str) -> str:
    """Extract candidate name using LLM"""
    NAME_PROMPT = """
    Extract the candidate's full name from the following resume text. 
    Return ONLY the name in JSON format like {"name": "John Doe"}. 
    If no name is found, return {"name": "Unknown"}.
    
    Resume Text:
    """
    
    try:
        truncated_text = resume_text[:2000]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": NAME_PROMPT + truncated_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("name", "Unknown")
    except Exception as e:
        print(f"Name extraction error: {str(e)}")
        return "Unknown"

class Candidate(BaseModel):
    id: str
    name: str
    fitScore: float
    overall_similarity: float
    llm_fit_score: float
    total_experience: float
    skills: Dict[str, List[str]]
    education_highlights: str
    experience_highlights: str
    summary: str
    justification: str
    email: str
    mobile_number: str
    resume_content: str

# New response model for ranking endpoint
class RankingResponse(BaseModel):
    run_id: str
    candidates: List[Candidate]

# --- Recursive Zip Processing ---
def process_zip_file(z: zipfile.ZipFile):
    """Recursively process all PDF files in a zip archive"""
    pdf_files = []
    for entry in z.namelist():
        if entry.endswith('/') or '.' not in entry:
            continue
            
        normalized = entry.lower().replace('\\', '/')
        if normalized.endswith('.pdf'):
            try:
                with z.open(entry) as file:
                    content = file.read()
                    pdf_files.append((entry, content))
            except Exception as e:
                print(f"      Error reading {entry}: {str(e)}")
    return pdf_files

# --- Database Helpers ---
def store_resume(user_id: str, batch_id: str, file_name: str, file_type: str, 
                 content: str, embedding: list, candidate_name: str) -> str:
    resume_doc = {
        "user_id": user_id,
        "batch_id": batch_id,
        "file_name": file_name,
        "file_type": file_type,
        "content": content,
        "embedding": embedding,
        "candidate_name": candidate_name,
        "created_at": datetime.utcnow()
    }
    result = get_resumes_collection().insert_one(resume_doc)
    return str(result.inserted_id)

def create_job_detail(user_id: str, job_role: str, job_description: str) -> str:
    job_doc = {
        "user_id": user_id,
        "job_role": job_role,
        "job_description": job_description,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = get_job_details_collection().insert_one(job_doc)
    return str(result.inserted_id)

def create_batch(user_id: str, job_details_id: str, resume_ids: List[str]) -> str:
    batch_doc = {
        "user_id": user_id,
        "job_details_id": job_details_id,
        "resumes": resume_ids,
        "created_at": datetime.utcnow()
    }
    result = get_batches_collection().insert_one(batch_doc)
    return str(result.inserted_id)

def store_screening_run(user_id: str, job_details_id: str, batch_id: str, 
                        run_start: datetime, run_end: datetime, candidates: List[dict]) -> str:
    run_doc = {
        "user_id": user_id,
        "job_details_id": job_details_id,
        "batch_id": batch_id,
        "run_start_time": run_start,
        "run_end_time": run_end,
        "candidates": candidates,
        "created_at": datetime.utcnow()
    }
    result = get_screening_runs_collection().insert_one(run_doc)
    return str(result.inserted_id)

# --- API Endpoint ---
@router.post("/", response_model=RankingResponse)
async def rank_and_parse_resumes(
    user_id: str = Form(...),
    job_role: str = Form(""),
    job_desc: str = Form(...),
    files: List[UploadFile] = File(...)
):
    total_start = datetime.utcnow()
    print(f"\n{'='*80}")
    print("STARTING RESUME SCREENING PROCESS")
    print(f"{'='*80}")
    
    # Create job detail
    job_details_id = create_job_detail(user_id, job_role, job_desc)
    log_activity(user_id, "job_created", f"Created job: {job_role}", job_details_id)
    
    # Phase 1: File Processing
    print("\n[PHASE 1] PROCESSING UPLOADED FILES")
    file_start = time.time()
    candidate_data = []  # (filename, resume_text, contact, file_bytes)
    resume_ids = []
    num_files = len(files)
    num_pdfs = 0
    
    for i, f in enumerate(files):
        print(f"  Processing file {i+1}/{num_files}: {f.filename}")
        content = await f.read()
        
        if f.filename.lower().endswith('.zip'):
            print(f"    Detected ZIP archive, extracting PDFs...")
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    pdf_files = process_zip_file(z)
                    print(f"      Found {len(pdf_files)} PDFs in archive")
                    
                    for filename, file_content in pdf_files:
                        try:
                            print(f"      Processing PDF: {filename}")
                            resume_text = extract_text_from_pdf_bytes(file_content)
                            if not resume_text.strip():
                                print(f"        Warning: Empty PDF content for {filename}")
                                continue
                                
                            contact = extract_contact_details(resume_text)
                            candidate_data.append((filename, resume_text, contact, file_content))
                            num_pdfs += 1
                        except Exception as e:
                            print(f"        Error processing {filename}: {str(e)}")
            except Exception as e:
                print(f"    Error processing ZIP file: {str(e)}")
        elif f.filename.lower().endswith('.pdf'):
            try:
                print(f"    Processing PDF file")
                resume_text = extract_text_from_pdf_bytes(content)
                if not resume_text.strip():
                    print(f"      Warning: Empty PDF content for {f.filename}")
                    continue
                    
                contact = extract_contact_details(resume_text)
                candidate_data.append((f.filename, resume_text, contact, content))
                num_pdfs += 1
            except Exception as e:
                print(f"    Error processing PDF: {str(e)}")
    
    if not candidate_data:
        print("\nERROR: No valid PDFs found in uploaded files")
        raise HTTPException(400, "No valid PDFs found.")
    
    # Create batch and store resumes
    for i, (filename, resume_text, contact, file_bytes) in enumerate(candidate_data):
        # Extract name later to avoid unnecessary LLM calls
        resume_id = store_resume(
            user_id=user_id,
            batch_id="",  # Will update later
            file_name=filename,
            file_type="pdf",
            content=resume_text,
            embedding=[],  # Will add after calculation
            candidate_name="Pending"
        )
        resume_ids.append(resume_id)
        candidate_data[i] = (*candidate_data[i], resume_id)
    
    # Create batch with resume IDs
    batch_id = create_batch(user_id, job_details_id, resume_ids)
    log_activity(user_id, "batch_created", f"Created batch with {len(resume_ids)} resumes", batch_id)
    
    # Update resumes with batch ID
    for resume_id in resume_ids:
        get_resumes_collection().update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"batch_id": batch_id}}
        )
    
    file_time = time.time() - file_start
    print(f"\n[PHASE 1 COMPLETE] Processed {num_pdfs} PDFs in {file_time:.2f} seconds")
    
    # Phase 2: Initial Screening
    print(f"\n[PHASE 2] INITIAL SCREENING")
    screen_start = time.time()
    print(f"  Encoding job description...")
    job_desc_emb = bi_encoder.encode(job_desc, convert_to_tensor=True, device=DEVICE)
    
    print(f"  Calculating similarity for {len(candidate_data)} candidates...")
    embeddings = []
    for i, (filename, resume_text, contact, file_bytes, resume_id) in enumerate(candidate_data):
        resume_emb = bi_encoder.encode(resume_text, convert_to_tensor=True, device=DEVICE)
        similarity = util.cos_sim(job_desc_emb, resume_emb).item()
        embeddings.append((resume_id, resume_emb.cpu().numpy().tolist()))
        candidate_data[i] = (*candidate_data[i], similarity)
    
    # Store embeddings
    for resume_id, embedding in embeddings:
        get_resumes_collection().update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"embedding": embedding}}
        )
    
    # Sort by initial similarity
    candidate_data.sort(key=lambda x: x[5], reverse=True)
    top_20 = candidate_data[:20]
    screen_time = time.time() - screen_start
    print(f"\n[PHASE 2 COMPLETE] Top 20 candidates selected in {screen_time:.2f} seconds")
    
    # Phase 3: LLM Analysis
    print(f"\n[PHASE 3] DETAILED LLM ANALYSIS")
    llm_start = time.time()
    detailed_candidates = []
    
    print(f"  Analyzing top 20 candidates with LLM...")
    for i, (filename, resume_text, contact, file_bytes, resume_id, overall_sim) in enumerate(top_20):
        # Extract name using LLM
        name = extract_name_with_llm(resume_text)
        
        # Update resume with name
        get_resumes_collection().update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"candidate_name": name}}
        )
        
        print(f"    Analyzing candidate {i+1}/20: {name} ({filename})")
        print(f"      Initial similarity: {overall_sim:.3f}")
        
        # Get detailed analysis from LLM
        analysis = analyze_with_llm(job_desc, resume_text)
        fit_score = analysis.get("fit_score", 0) / 100.0
        print(f"      LLM fit score: {fit_score:.3f}")
        
        detailed_candidates.append({
            "resume_id": resume_id,
            "filename": filename,
            "name": name,
            "resume_text": resume_text,
            "contact": contact,
            "overall_sim": overall_sim,
            "llm_analysis": analysis,
            "llm_fit_score": fit_score
        })
    
    # Sort by LLM fit score
    detailed_candidates.sort(key=lambda x: x["llm_fit_score"], reverse=True)
    top_10 = detailed_candidates[:10]
    llm_time = time.time() - llm_start
    print(f"\n[PHASE 3 COMPLETE] LLM analysis completed in {llm_time:.2f} seconds")
    
    # Prepare final response and screening run data
    print(f"\n[PHASE 4] PREPARING FINAL RESULTS")
    final_results = []
    screening_candidates = []
    
    for i, candidate in enumerate(top_10):
        analysis = candidate["llm_analysis"]
        final_candidate = Candidate(
            id=candidate["resume_id"],
            name=candidate["name"],
            fitScore=round(candidate["llm_fit_score"] * 100, 1),
            overall_similarity=round(candidate["overall_sim"], 4),
            llm_fit_score=round(candidate["llm_fit_score"] * 100, 1),
            total_experience=0,  # Placeholder
            skills={
                "exact_matches": analysis["technical_skills"].get("exact_matches", []),
                "transferable": analysis["technical_skills"].get("transferable_skills", []),
                "non_technical": analysis.get("non_technical_skills", [])
            },
            education_highlights=analysis.get("education_highlights", ""),
            experience_highlights=analysis.get("experience_highlights", ""),
            summary=analysis.get("overall_summary", ""),
            justification=analysis.get("justification", ""),
            email=candidate["contact"]["email"],
            mobile_number=candidate["contact"]["mobile_number"],
            resume_content=candidate["resume_text"]
        )
        final_results.append(final_candidate)
        
        # Prepare for screening run storage
        screening_candidate = {
            "resume_id": candidate["resume_id"],
            "candidate_name": candidate["name"],
            "batch_id": batch_id,
            "file_name": candidate["filename"],
            "file_type": "pdf",
            "ai_fit_score": final_candidate.fitScore,
            "skill_similarity": candidate["overall_sim"],
            "candidate_summary": final_candidate.summary,
            "skill_assessment": {
                "exact_matches": final_candidate.skills["exact_matches"],
                "transferable_skills": final_candidate.skills["transferable"],
                "non_technical_skills": final_candidate.skills["non_technical"]
            },
            "experience_highlights": final_candidate.experience_highlights,
            "education_highlights": final_candidate.education_highlights,
            "gaps": analysis.get("gaps", []),
            "ai_justification": final_candidate.justification,
            "resume_content_preview": candidate["resume_text"][:1000],
            "questions_generated": False,
            "generated_questions": [],
            "alternate_candidate_searched": False,
            "alternate_candidate": {}
        }
        screening_candidates.append(screening_candidate)
        print(f"  Prepared candidate {i+1}: {candidate['name']} - Fit: {candidate['llm_fit_score']:.3f}")
    
    # Store screening run
    run_id = store_screening_run(
        user_id=user_id,
        job_details_id=job_details_id,
        batch_id=batch_id,
        run_start=total_start,
        run_end=datetime.utcnow(),
        candidates=screening_candidates
    )
    log_activity(user_id, "screening_run", f"Screening run completed for {len(candidate_data)} candidates", run_id)
    
    total_time = time.time() - total_start.timestamp()
    
    # Performance summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total candidates processed: {len(candidate_data)}")
    print(f"Files processed: {num_files} ({num_pdfs} PDFs extracted)")
    print(f"Initial screening time: {screen_time:.2f} seconds")
    print(f"LLM analysis time: {llm_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Top candidate: {top_10[0]['name']} - Fit: {top_10[0]['llm_fit_score']:.3f}")
    print(f"{'='*80}")
    
    return {
        "run_id": run_id,
        "candidates": final_results
    }

# --- LLM Analysis ---
def analyze_with_llm(jd_text: str, resume_text: str) -> Dict:
    """Analyze candidate with LLM to compute detailed fit score"""
    SYSTEM_PROMPT = """
    You are an expert HR analyst. Analyze a candidate's resume against a job description and provide:
    1. Overall summary (1-2 sentences)
    2. Comprehensive fit score (0-100%) based on:
       - Technical skills (exact matches and transferable skills)
       - Non-technical skills
       - Experience relevance
       - Education qualifications
    3. If skills are similar, assign fit score based on how closely the skills match the job requirements
    4. Skill highlights (technical and non-technical)
    5. Experience highlights
    6. Education highlights
    7. Justification for the fit score
    8. You can also assign a low score if the skills are not in line with the job description
    
    Output format (JSON):
    {
        "overall_summary": "",
        "fit_score": 0,
        "technical_skills": {
            "exact_matches": [],
            "transferable_skills": []
        },
        "non_technical_skills": [],
        "experience_highlights": "",
        "education_highlights": "",
        "justification": "",
        "gaps": []
    }
    
    Guidelines:
    - Be objective and critical
    - Transferable skills: show how non-direct experience could be valuable
    - Fit score: percentage reflecting overall suitability
    - Highlight most relevant qualifications
    """
    
    try:
        print(f"  Sending analysis request to LLM...")
        start_time = time.time()
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Job Description:\n{jd_text}\n\nResume:\n{resume_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        elapsed = time.time() - start_time
        print(f"  LLM analysis completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        print(f"  LLM Error: {str(e)}")
        return {
            "overall_summary": "Analysis failed",
            "fit_score": 0,
            "technical_skills": {"exact_matches": [], "transferable_skills": []},
            "non_technical_skills": [],
            "experience_highlights": "",
            "education_highlights": "",
            "justification": "",
            "gaps": []
        }
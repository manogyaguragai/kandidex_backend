import re
import io
import zipfile
import fitz
import torch
import time
import json
import asyncio
from typing import List, Dict, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from openai import AsyncOpenAI
import os
from datetime import datetime
from config import (
    get_job_details_collection,
    get_resumes_collection,
    get_batches_collection,
    get_screening_runs_collection,
    get_settings_collection,
    log_activity
)
from bson import ObjectId

router = APIRouter(prefix="/rank", tags=["ranking"])

# Initialize OpenAI client
aopenai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Candidate(BaseModel):
    id: str
    name: str
    file_name: str
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

class RankingResponse(BaseModel):
    run_id: str
    user_id: str
    candidates: List[Candidate]

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

# --- Async LLM Functions ---
async def extract_name_with_llm(resume_text: str) -> str:
    """Extract candidate name using LLM (async)"""
    NAME_PROMPT = """
    Extract the candidate's full name from the following resume text. 
    Return ONLY the name in JSON format like {"name": "John Doe"}. 
    If no name is found, return {"name": "Unknown"}.
    """
    
    try:
        truncated_text = resume_text[:2000]
        response = await aopenai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": NAME_PROMPT},
                {"role": "user", "content": truncated_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("name", "Unknown")
    except Exception as e:
        print(f"Name extraction error: {str(e)}")
        return "Unknown"

async def extract_names_with_llm_batch(resume_texts: List[str]) -> List[str]:
    """Batch extract candidate names using LLM"""
    tasks = [extract_name_with_llm(text) for text in resume_texts]
    return await asyncio.gather(*tasks)

async def analyze_one_resume_with_llm(jd_text: str, resume_text: str) -> Dict:
    """Analyze one candidate with LLM (async)"""
    SYSTEM_PROMPT = """
    You are an expert HR analyst. Analyze a candidate's resume against a job description and provide:
    1. Overall summary (1-2 sentences)
    2. Comprehensive fit score (0-100%) 
    3. Skill highlights (technical and non-technical)
    4. Experience highlights
    5. Education highlights
    6. Justification for the fit score
    7. List of identified skill/experience gaps
    
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
    """
    
    try:
        response = await aopenai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Job Description:\n{jd_text}\n\nResume:\n{resume_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {str(e)}")
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

async def analyze_with_llm_batch(jd_text: str, resume_texts: List[str]) -> List[Dict]:
    """Batch analyze candidates with LLM"""
    tasks = [analyze_one_resume_with_llm(jd_text, text) for text in resume_texts]
    return await asyncio.gather(*tasks)

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
        "created_at": datetime.now()
    }
    result = get_resumes_collection().insert_one(resume_doc)
    return str(result.inserted_id)

def create_job_detail(user_id: str, job_role: str, job_description: str) -> str:
    job_doc = {
        "user_id": user_id,
        "job_role": job_role,
        "job_description": job_description,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    result = get_job_details_collection().insert_one(job_doc)
    return str(result.inserted_id)

def create_batch(user_id: str, job_details_id: str, resume_ids: List[str]) -> str:
    batch_doc = {
        "user_id": user_id,
        "job_details_id": job_details_id,
        "resumes": resume_ids,
        "created_at": datetime.now()
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
        "created_at": datetime.now()
    }
    result = get_screening_runs_collection().insert_one(run_doc)
    return str(result.inserted_id)

# --- Device Handling ---
def get_device():
    """Dynamically determine the best available device with error handling"""
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            print("CUDA is available - using GPU")
            # Test a small operation to verify functionality
            test_tensor = torch.tensor([1.0]).cuda()
            if test_tensor.device.type == 'cuda':
                print("GPU operations verified")
                return "cuda"
            else:
                print("CUDA test failed - falling back to CPU")
                return "cpu"
        return "cpu"
    except Exception as e:
        print(f"CUDA initialization failed: {str(e)} - Using CPU")
        return "cpu"

# --- API Endpoint ---
@router.post("/", response_model=RankingResponse)
async def rank_and_parse_resumes(
    user_id: str = Form(...),
    job_role: str = Form(""),
    job_desc: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Dynamically determine device at runtime
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    
    # Initialize models AFTER device determination
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    bi_encoder.max_seq_length = 512

    # Fetch user settings for phase ranking numbers
    user_settings = get_settings_collection().find_one({"user_id": user_id})
    
    # Set default values if settings not found
    phase1_limit = user_settings.get("phase1_ranking_number", 20) if user_settings else 20
    phase2_limit = user_settings.get("phase2_ranking_number", 10) if user_settings else 10
    
    # Validate limits
    if phase1_limit <= 0 or phase2_limit <= 0:
        raise HTTPException(400, "Ranking numbers must be positive values")
    if phase1_limit < phase2_limit:
        raise HTTPException(400, "Phase1 limit must be greater than or equal to Phase2 limit")
    
    total_start = datetime.now()
    print(f"\n{'='*80}")
    print(f"STARTING RESUME SCREENING PROCESS (Phase1: {phase1_limit}, Phase2: {phase2_limit})")
    print(f"{'='*80}")
    
    # Create job detail
    job_details_id = create_job_detail(user_id, job_role, job_desc)
    log_activity(user_id, "job_created", f"Created job: {job_role}", job_details_id)
    
    # Phase 1: File Processing
    print("\n[PHASE 1] PROCESSING UPLOADED FILES")
    file_start = time.time()
    candidate_data = []  # (filename, resume_text, contact, resume_id)
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
                            candidate_data.append((filename, resume_text, contact))
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
                candidate_data.append((f.filename, resume_text, contact))
                num_pdfs += 1
            except Exception as e:
                print(f"    Error processing PDF: {str(e)}")
    
    if not candidate_data:
        print("\nERROR: No valid PDFs found in uploaded files")
        raise HTTPException(400, "No valid PDFs found.")
    
    # Create batch and store resumes
    for i, (filename, resume_text, contact) in enumerate(candidate_data):
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
        candidate_data[i] = (filename, resume_text, contact, resume_id)
    
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
    for i, (filename, resume_text, contact, resume_id) in enumerate(candidate_data):
        resume_emb = bi_encoder.encode(resume_text, convert_to_tensor=True, device=DEVICE)
        similarity = util.cos_sim(job_desc_emb, resume_emb).item()
        embeddings.append((resume_id, resume_emb.cpu().numpy().tolist()))
        candidate_data[i] = (filename, resume_text, contact, resume_id, similarity)
    
    # Store embeddings
    for resume_id, embedding in embeddings:
        get_resumes_collection().update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"embedding": embedding}}
        )
    
    # Sort by initial similarity
    candidate_data.sort(key=lambda x: x[4], reverse=True)  # Index 4 is similarity
    
    # Use phase1_limit instead of hardcoded 20
    topp1 = candidate_data[:phase1_limit]
    print(f"Selected top {phase1_limit} candidates based on similarity:\n")
    for candidate in topp1:
        filename, resume_text, contact, resume_id, similarity = candidate
        print(f"  {filename} | Similarity: {similarity*100:.2f}%")
    
    screen_time = time.time() - screen_start
    print(f"\n[PHASE 2 COMPLETE] Top {phase1_limit} candidates selected in {screen_time:.2f} seconds")
    
    # Phase 3: Async LLM Processing
    print(f"\n[PHASE 3] ASYNC LLM PROCESSING")
    llm_start = time.time()
    
    # Prepare batch data for LLM - use topp1 instead of top_20
    resume_texts_topp1 = [candidate[1] for candidate in topp1]  # Index 1 is resume_text
    
    # Batch name extraction
    print("  Starting batch name extraction...")
    name_start = time.time()
    names = await extract_names_with_llm_batch(resume_texts_topp1)
    name_time = time.time() - name_start
    print(f"  Batch name extraction completed in {name_time:.2f} seconds")
    
    # Update database with names
    for i, (candidate, name) in enumerate(zip(topp1, names)):
        filename, resume_text, contact, resume_id, similarity = candidate
        get_resumes_collection().update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"candidate_name": name}}
        )
        topp1[i] = (filename, resume_text, contact, resume_id, similarity, name)
    
    # Batch detailed analysis
    print("  Starting batch detailed analysis...")
    analysis_start = time.time()
    analyses = await analyze_with_llm_batch(job_desc, resume_texts_topp1)
    analysis_time = time.time() - analysis_start
    print(f"  Batch analysis completed in {analysis_time:.2f} seconds")
    
    # Process results
    detailed_candidates = []
    for i, candidate in enumerate(topp1):
        filename, resume_text, contact, resume_id, similarity, name = candidate
        analysis = analyses[i]
        fit_score = analysis.get("fit_score", 0) / 100.0
        
        detailed_candidates.append({
            "resume_id": resume_id,
            "filename": filename,
            "name": name,
            "resume_text": resume_text,
            "contact": contact,
            "overall_sim": similarity,
            "llm_analysis": analysis,
            "llm_fit_score": fit_score
        })
    
    # Sort by LLM fit score
    detailed_candidates.sort(key=lambda x: x["llm_fit_score"], reverse=True)
    
    # Use phase2_limit instead of hardcoded 10
    topp2 = detailed_candidates[:phase2_limit]
    llm_time = time.time() - llm_start
    print(f"\n[PHASE 3 COMPLETE] LLM processing completed in {llm_time:.2f} seconds")
    
    # Prepare final response and screening run data
    print(f"\n[PHASE 4] PREPARING FINAL RESULTS")
    final_results = []
    screening_candidates = []
    
    # Use topp2 instead of top_10
    for i, candidate in enumerate(topp2):
        analysis = candidate["llm_analysis"]
        final_candidate = Candidate(
            id=candidate["resume_id"],
            name=candidate["name"],
            file_name=candidate["filename"],
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
        run_end=datetime.now(),
        candidates=screening_candidates
    )
    log_activity(user_id, "screening_run", 
                 f"Screening run completed for {len(candidate_data)} candidates (Phase1: {phase1_limit}, Phase2: {phase2_limit})", 
                 run_id)
    
    total_time = (datetime.now() - total_start).total_seconds()
    
    # Performance summary - updated with dynamic limits
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total candidates processed: {len(candidate_data)}")
    print(f"Files processed: {num_files} ({num_pdfs} PDFs extracted)")
    print(f"Initial screening time: {screen_time:.2f} seconds")
    print(f"LLM processing time: {llm_time:.2f} seconds (Names: {name_time:.2f}s, Analysis: {analysis_time:.2f}s)")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Phase1 candidates: {phase1_limit}")
    print(f"Phase2 candidates: {phase2_limit}")
    print(f"Top candidate: {topp2[0]['name']} - Fit: {topp2[0]['llm_fit_score']:.3f}")
    print(f"{'='*80}")
    
    results = {
        "run_id": run_id,
        "user_id": user_id,
        "candidates": final_results
    }
    
    return results
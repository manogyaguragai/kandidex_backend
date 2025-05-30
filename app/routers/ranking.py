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
from pathlib import Path
from openai import OpenAI
import os

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
        # Use the first 2000 characters to stay within token limits
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

# --- Recursive Zip Processing ---
def process_zip_file(z: zipfile.ZipFile):
    """Recursively process all PDF files in a zip archive"""
    pdf_files = []
    for entry in z.namelist():
        # Skip directories (both with and without trailing slash)
        if entry.endswith('/') or '.' not in entry:
            continue
            
        # Normalize path and check extension
        normalized = entry.lower().replace('\\', '/')
        if normalized.endswith('.pdf'):
            try:
                with z.open(entry) as file:
                    content = file.read()
                    pdf_files.append((entry, content))
            except Exception as e:
                print(f"      Error reading {entry}: {str(e)}")
    return pdf_files

# --- API Endpoint ---
@router.post("/", response_model=List[Candidate])
async def rank_and_parse_resumes(
    job_desc: str = Form(...),
    files: List[UploadFile] = File(...)
):
    total_start = time.time()
    print(f"\n{'='*80}")
    print("STARTING RESUME SCREENING PROCESS")
    print(f"{'='*80}")
    
    # Phase 1: File Processing
    print("\n[PHASE 1] PROCESSING UPLOADED FILES")
    file_start = time.time()
    candidate_data = []  # (filename, resume_text, contact)
    num_files = len(files)
    num_pdfs = 0
    
    # Process all files and extract text
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
                            candidate_data.append((
                                filename,
                                resume_text,
                                contact
                            ))
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
                candidate_data.append((
                    f.filename,
                    resume_text,
                    contact
                ))
                num_pdfs += 1
            except Exception as e:
                print(f"    Error processing PDF: {str(e)}")
    
    if not candidate_data:
        print("\nERROR: No valid PDFs found in uploaded files")
        raise HTTPException(400, "No valid PDFs found.")
    
    file_time = time.time() - file_start
    print(f"\n[PHASE 1 COMPLETE] Processed {num_pdfs} PDFs in {file_time:.2f} seconds")
    
    # Phase 2: Initial Screening
    print(f"\n[PHASE 2] INITIAL SCREENING")
    screen_start = time.time()
    print(f"  Encoding job description...")
    job_desc_emb = bi_encoder.encode(job_desc, convert_to_tensor=True, device=DEVICE)
    
    print(f"  Calculating similarity for {len(candidate_data)} candidates...")
    # Calculate similarities
    for i, (filename, resume_text, contact) in enumerate(candidate_data):
        resume_emb = bi_encoder.encode(resume_text, convert_to_tensor=True, device=DEVICE)
        similarity = util.cos_sim(job_desc_emb, resume_emb).item()
        # Add similarity score to candidate data
        candidate_data[i] = (*candidate_data[i], similarity)
    
    # Sort by initial similarity
    candidate_data.sort(key=lambda x: x[3], reverse=True)
    top_15 = candidate_data[:15]
    screen_time = time.time() - screen_start
    print(f"\n[PHASE 2 COMPLETE] Top 15 candidates selected in {screen_time:.2f} seconds")
    
    # Phase 3: LLM Analysis
    print(f"\n[PHASE 3] DETAILED LLM ANALYSIS")
    llm_start = time.time()
    detailed_candidates = []
    
    print(f"  Analyzing top 15 candidates with LLM...")
    for i, (filename, resume_text, contact, overall_sim) in enumerate(top_15):
        # Extract name using LLM
        name = extract_name_with_llm(resume_text)
        print(f"    Analyzing candidate {i+1}/15: {name} ({filename})")
        print(f"      Initial similarity: {overall_sim:.3f}")
        
        # Get detailed analysis from LLM
        analysis = analyze_with_llm(job_desc, resume_text)
        fit_score = analysis.get("fit_score", 0) / 100.0
        print(f"      LLM fit score: {fit_score:.3f}")
        
        detailed_candidates.append({
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
    
    # Prepare final response
    print(f"\n[PHASE 4] PREPARING FINAL RESULTS")
    final_results = []
    for i, candidate in enumerate(top_10):
        analysis = candidate["llm_analysis"]
        final_results.append(Candidate(
            id=candidate["filename"],
            name=candidate["name"],
            fitScore=round(candidate["llm_fit_score"] * 100, 1),  # Convert to percentage with 1 decimal
            overall_similarity=round(candidate["overall_sim"], 4),
            llm_fit_score=round(candidate["llm_fit_score"] * 100, 1),  # Percentage for display
            total_experience=0,
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
            mobile_number=candidate["contact"]["mobile_number"]
        ))
        print(f"  Prepared candidate {i+1}: {candidate['name']} - Fit: {candidate['llm_fit_score']:.3f}")
    
    total_time = time.time() - total_start
    
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
    
    return final_results

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
        "justification": ""
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
            "justification": ""
        }
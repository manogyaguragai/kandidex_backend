import re
import io
import zipfile
import fitz
import torch
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from pathlib import Path

router = APIRouter(prefix="/rank", tags=["ranking"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "output" / "output_bert_mini_job_resume"

bi_encoder = SentenceTransformer(str(MODEL_DIR), device=DEVICE)

bi_encoder.max_seq_length = 512
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=DEVICE
)

class Candidate(BaseModel):
    id: str
    name: str
    fitScore: float
    total_experience: float
    skills: List[str]
    education: List[str]
    email: str
    mobile_number: str
    summary: str

# --- Extraction Helpers ---

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes.
    """
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_contact_details(text: str) -> Dict[str, str]:
    """
    Extract email and phone via regex.
    """
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)
    return {"email": emails[0] if emails else "", "mobile_number": phones[0] if phones else ""}


def extract_section(text: str, section: str) -> str:
    """
    Dynamically extract a section by header name (e.g., 'Skills', 'Education').
    Captures text until the next all-caps heading or end of document.
    """
    pattern = rf"(?im)^{section}\s*[:\-]?\s*(.*?)\s*(?=^\S.+?:|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_skills(text: str) -> List[str]:
    """
    Extract skills list from 'Skills' section, splitting on commas or line breaks.
    Limit to 10 items max.
    """
    sec = extract_section(text, "Skills")
    items = re.split(r"[,\n]", sec)
    return [item.strip() for item in items if item.strip()][:10]


def extract_education(text: str) -> List[str]:
    """
    Extract education entries from 'Education' section.
    Limit to 10 items max.
    """
    sec = extract_section(text, "Education")
    items = re.split(r"[\n;]", sec)
    return [item.strip() for item in items if item.strip()][:10]


def extract_experience_years(text: str) -> float:
    """
    Extract experience details from 'Experience' section and sum year mentions.
    """
    sec = extract_section(text, "Experience")
    # Sum patterns like '2 years', '6 mos', '1.5 yrs'
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|mos?|months?)", sec, flags=re.IGNORECASE)
    return sum(float(m) for m in matches)


def parse_resume_bytes(file_bytes: bytes) -> Dict:
    """
    Parse resume PDF bytes into structured info.
    """
    text = extract_text_from_pdf_bytes(file_bytes)
    contact = extract_contact_details(text)
    # Dynamic sections
    skills = extract_skills(text)
    education = extract_education(text)
    total_exp = extract_experience_years(text)

    return {
        "name": text.splitlines()[0].strip(),
        "email": contact["email"],
        "mobile_number": contact["mobile_number"],
        "skills": skills,
        "education": education,
        "total_experience": total_exp,
        "summary": text[:300].replace("\n", " ") + "…"
    }

# --- API Endpoint ---
@router.post("/", response_model=List[Candidate])
async def rank_and_parse_resumes(
    job_desc: str = Form(...),
    files: List[UploadFile] = File(...)
):
    entries = []  # (filename, info, embedding)

    async def process_pdf(name: str, content: bytes):
        info = parse_resume_bytes(content)
        emb = bi_encoder.encode(info["summary"], convert_to_tensor=True, device=DEVICE)
        entries.append((name, info, emb))

    # Read and parse files
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for entry in z.namelist():
                    if entry.lower().endswith('.pdf'):
                        await process_pdf(entry, z.read(entry))
        elif f.filename.lower().endswith('.pdf'):
            await process_pdf(f.filename, content)
        else:
            raise HTTPException(400, f"Unsupported file {f.filename}")

    if not entries:
        raise HTTPException(400, "No valid PDFs found.")

    # Semantic ranking
    query_emb = bi_encoder.encode(job_desc, convert_to_tensor=True, device=DEVICE)
    corpus = torch.stack([e[2] for e in entries], dim=0).to(DEVICE)
    hits = util.semantic_search(query_emb, corpus, top_k=len(entries))[0]

    # Re-rank
    cross_inp = [(job_desc, entries[h['corpus_id']][1]['summary']) for h in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    for i, score in enumerate(cross_scores):
        hits[i]['cross_score'] = float(score)

    # Top 10
    top_hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)[:10]
    results = []
    for h in top_hits:
        idx = h['corpus_id']
        name, info, _ = entries[idx]
        results.append(Candidate(
            id=name,
            name=info['name'],
            fitScore=h['cross_score'],
            total_experience=info['total_experience'],
            skills=info['skills'],
            education=info['education'],
            email=info['email'],
            mobile_number=info['mobile_number'],
            summary=info['summary']
        ))
        
    return results

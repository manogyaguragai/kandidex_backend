import os
import re
import fitz  # PyMuPDF
import spacy
from typing import List, Dict

# Load SpaCy model once

nlp = spacy.load("en_core_web_sm")

# List of known skills (extendable)
KNOWN_SKILLS = [
    "python", "java", "sql", "excel", "c++", "machine learning",
    "deep learning", "nlp", "flask", "django", "react", "node", "aws"
]


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF.
    """
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    return text.strip()


def extract_contact_details(text: str) -> Dict[str, str]:
    """
    Extracts email and mobile number via regex.
    """
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r'\+?\d[\d\s().-]{8,}\d', text)
    return {
        "email": email[0] if email else "",
        "mobile_number": phone[0] if phone else ""
    }


def extract_skills(text: str, known_skills: List[str] = KNOWN_SKILLS) -> List[str]:
    """
    Matches known skills in resume text using spaCy.
    """
    doc = nlp(text.lower())
    found = {token.text for token in doc if token.text in known_skills}
    return list(found)


def extract_education(text: str) -> List[str]:
    """
    Finds degree keywords in resume text.
    """
    degrees = re.findall(
        r'(?:B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|Bachelor|Master|Ph\.?D)[^,\n]*',
        text, flags=re.IGNORECASE)
    return list({deg.strip() for deg in degrees})


def extract_experience_years(text: str) -> float:
    """
    Estimates total years of experience by summing year mentions.
    """
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text, flags=re.IGNORECASE)
    years = sum(float(m) for m in matches)
    return years


def extract_company_names(text: str) -> List[str]:
    """
    Naive extraction of company names by 'at XYZ'.
    """
    comps = re.findall(r'at\s+([A-Z][A-Za-z&\- ]+)', text)
    return list({c.strip() for c in comps})


def parse_resume_bytes(file_bytes: bytes) -> Dict:
    """
    Full parsing pipeline: text extraction + structured field extraction.
    """
    text = extract_text_from_pdf_bytes(file_bytes)
    contact = extract_contact_details(text)
    skills = extract_skills(text)
    education = extract_education(text)
    experience_years = extract_experience_years(text)
    companies = extract_company_names(text)

    return {
        "name": text.split('\n')[0].strip(),
        "email": contact["email"],
        "mobile_number": contact["mobile_number"],
        "skills": skills,
        "education": education,
        "total_experience": experience_years,
        "company_names": companies,
        "summary": text[:300].replace("\n", " ") + "…"
    }

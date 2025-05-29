import os
import spacy
from spacy.matcher import Matcher
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

class ResumeParser:
    def __init__(self):
        # Automatically download spaCy model if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        self.matcher = Matcher(self.nlp.vocab)
        self._add_section_patterns()
    
    def _add_section_patterns(self):
        # Define patterns for common resume sections
        section_patterns = {
            "SKILLS": [
                [{"LOWER": "skills"}],
                [{"LOWER": "technical"}, {"LOWER": "skills"}],
                [{"LOWER": "key"}, {"LOWER": "competencies"}],
                [{"LOWER": "core"}, {"LOWER": "competencies"}],
                [{"LOWER": "technical"}, {"LOWER": "expertise"}]
            ],
            "EXPERIENCE": [
                [{"LOWER": "experience"}],
                [{"LOWER": "work"}, {"LOWER": "history"}],
                [{"LOWER": "employment"}, {"LOWER": "history"}],
                [{"LOWER": "professional"}, {"LOWER": "experience"}],
                [{"LOWER": "work"}, {"LOWER": "experience"}]
            ],
            "EDUCATION": [
                [{"LOWER": "education"}],
                [{"LOWER": "academic"}, {"LOWER": "background"}],
                [{"LOWER": "academic"}, {"LOWER": "qualifications"}],
                [{"LOWER": "degrees"}],
                [{"LOWER": "certifications"}]
            ]
        }
        
        for section_name, patterns in section_patterns.items():
            self.matcher.add(section_name.upper(), patterns)
    
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            return ""
        return text
    
    def extract_sections(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        # Find section boundaries
        section_boundaries = {}
        for match_id, start, end in matches:
            section_name = self.nlp.vocab.strings[match_id]
            # Only keep the first occurrence of each section
            if section_name not in section_boundaries:
                section_boundaries[section_name] = (start, end)
        
        # Sort sections by their position in the document
        sorted_sections = sorted(section_boundaries.items(), key=lambda x: x[1][0])
        
        # Extract section content
        sections = {}
        full_text = text
        for i, (section_name, (start, end)) in enumerate(sorted_sections):
            section_start = doc[end].idx + len(doc[end].text)
            if i < len(sorted_sections) - 1:
                next_section_start = sorted_sections[i+1][1][0]
                section_end = doc[next_section_start].idx
            else:
                section_end = len(text)
            
            content = text[section_start:section_end].strip()
            sections[section_name] = content
        
        # If we didn't find any sections, use full text
        if not sections:
            sections["FULL_TEXT"] = text
        else:
            sections["FULL_TEXT"] = full_text
        
        return sections

class ResumeScreener:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.parser = ResumeParser()
        # Use a smaller model by default for faster processing
        self.model = SentenceTransformer(model_name)
        self.weights = {
            "overall": 0.4,
            "skills": 0.4,
            "experience": 0.2
        }
        self.thresholds = {
            "min_skill": 0.25,
            "min_overall": 0.3
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        return ' '.join(text.split()).lower()
    
    def calculate_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def extract_experience_years(self, text):
      """Extract total years of experience from text with boundary checks"""
      if not text:
          return 0.0
          
      doc = self.parser.nlp(text)
      total_years = 0.0
      
      # Pattern 1: Explicit year durations ("5 years")
      for i, token in enumerate(doc):
          # Check if there's a next token and it's a year indicator
          if token.like_num and i < len(doc) - 1:
              next_token = doc[i+1]
              if next_token.text.lower() in ["years", "yrs", "year"]:
                  try:
                      total_years += float(token.text)
                  except ValueError:
                      pass
      
      # Pattern 2: Date ranges ("2018-2022")
      for ent in doc.ents:
          if ent.label_ == "DATE" and "-" in ent.text:
              parts = ent.text.split("-")
              if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                  total_years += (int(parts[1]) - int(parts[0]))
      
      # Pattern 3: Month durations ("18 months")
      for i, token in enumerate(doc):
          # Check if there's a next token and it's a month indicator
          if token.like_num and i < len(doc) - 1:
              next_token = doc[i+1]
              if next_token.text.lower() in ["months", "mos"]:
                  try:
                      months = float(token.text)
                      total_years += months / 12
                  except ValueError:
                      pass
      
      return total_years
    
    def calculate_match_score(self, resume_text, jd_text):
        """Calculate comprehensive match score"""
        # Parse resume
        resume = self.parser.extract_sections(resume_text)
        jd = self.parser.extract_sections(jd_text)
        
        # Extract texts with fallbacks
        resume_full = resume.get("FULL_TEXT", resume_text)
        jd_full = jd.get("FULL_TEXT", jd_text)
        resume_skills = resume.get("SKILLS", "")
        jd_skills = jd.get("SKILLS", "")
        resume_exp = resume.get("EXPERIENCE", "")
        jd_exp = jd.get("EXPERIENCE", "")
        
        # Calculate similarities
        overall_sim = self.calculate_similarity(resume_full, jd_full)
        skill_sim = self.calculate_similarity(resume_skills, jd_skills)
        
        # Experience matching
        jd_years = self.extract_experience_years(jd_exp) or 0
        resume_years = self.extract_experience_years(resume_exp) or 0
        
        if jd_years > 0:
            exp_match = min(resume_years / jd_years, 1.0)  # Cap at 100% match
        else:
            exp_match = 0.5  # Neutral if JD doesn't specify years
        
        # Calculate composite score
        fit_score = (
            self.weights["overall"] * overall_sim +
            self.weights["skills"] * skill_sim +
            self.weights["experience"] * exp_match
        )
        
        return {
            "overall_similarity": overall_sim,
            "skill_similarity": skill_sim,
            "experience_match": exp_match,
            "years_experience": resume_years,
            "fit_score": fit_score,
            "qualified": fit_score >= self.thresholds["min_overall"] and 
                        skill_sim >= self.thresholds["min_skill"]
        }
    
    def screen_resumes(self, jd_text, pdf_directory, top_n=10):
        """Process all PDFs in directory and return top matches"""
        results = []
        
        # Process each PDF
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                print(f"Processing: {filename}")
                resume_text = self.parser.extract_text_from_pdf(pdf_path)
                
                if resume_text:
                    try:
                        score = self.calculate_match_score(resume_text, jd_text)
                        results.append({
                            "filename": filename,
                            **score
                        })
                        print(f"  Score: {score['fit_score']:.3f}")
                    except Exception as e:
                        print(f"  Error processing resume: {str(e)}")
                else:
                    print(f"  Failed to extract text from PDF")
        
        # Sort and return top matches
        results.sort(key=lambda x: x["fit_score"], reverse=True)
        return results[:top_n]

if __name__ == "__main__":
    # Initialize screener
    screener = ResumeScreener()
    
    # Job description
    job_description = """
Position Details
Advertisement No.: 14/2081/082

Position: Medical Officer

Level/Category: Officer, Eighth Level

Minimum Qualification: MBBS from a recognized educational institution and registered with the concerned medical council

Type: Contract

Number of Positions: 3

Application Procedure and Submission Location
Application forms can be obtained from the Administrative Section of the hospital or downloaded from the hospital’s website: gulmihospital.lumbini.gov.np

Completed applications must be submitted to Gulmi Hospital along with the required documents.

Required Documents
Personal details of the applicant

Certified copies of academic qualifications

Certified copy of Nepali citizenship certificate

Certified copy of relevant work experience

Certified copy of registration with the respective medical council (as per prevailing Nepali laws)

All documents must be self-attested on the back

Three recent passport-sized photographs

All information must be submitted using the Public Service Commission application form format.

Application Fee
For Eighth Level: NPR 1,000

The fee must be deposited into the account of the Gulmi Hospital Development Committee.

Account No.: 3010100202070001

Bank: Rastriya Banijya Bank, Tamghas Branch

Application Deadline
Last date for submission: 2082/02/08 within office hours

If the deadline falls on a public holiday, applications will be accepted on the next working day

Selection Basis and Process
Shortlisting

Interview (Only shortlisted candidates will be called for the interview)

Contact Date and Location
The day after the application deadline

Location: Gulmi Hospital

Age Limit
Candidates must be at least 21 years old and not exceed 45 years of age

Job Responsibilities
As per the service and group-related duties applicable to the position
"""

    # Directory containing resumes
    resume_directory = "/home/manogyaguragai/Desktop/Resumes"
    
    # Get top matches
    top_matches = screener.screen_resumes(job_description, resume_directory, top_n=10)
    
    # Print results
    if top_matches:
        print("\nTOP MATCHES:")
        print("-" * 85)
        print(f"{'Filename':<25} | {'Overall':>7} | {'Skills':>7} | {'Exp Match':>9} | {'Yrs Exp':>6} | {'Score':>6}")
        print("-" * 85)
        for match in top_matches:
            print(
                f"{match['filename'][:24]:<25} | "
                f"{match['overall_similarity']:>7.3f} | "
                f"{match['skill_similarity']:>7.3f} | "
                f"{match['experience_match']:>9.3f} | "
                f"{match['years_experience']:>6.1f} | "
                f"{match['fit_score']:>6.3f}"
            )
    else:
        print("No resumes processed successfully")
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeScreener:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Use a small, efficient model
        self.model = SentenceTransformer(model_name)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass  # Silently handle errors
        return text
    
    def preprocess_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def calculate_fit_score(self, resume_text, jd_text):
        """Calculate semantic similarity as fit score"""
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
            
        emb1 = self.model.encode(resume_text)
        emb2 = self.model.encode(jd_text)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def screen_resumes(self, jd_text, pdf_directory, top_n=10):
        """Process PDFs and return top matches with fit scores"""
        results = []
        clean_jd = self.preprocess_text(jd_text)
        
        # Process each PDF
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                resume_text = self.extract_text_from_pdf(pdf_path)
                
                if resume_text:
                    try:
                        clean_resume = self.preprocess_text(resume_text)
                        fit_score = self.calculate_fit_score(clean_resume, clean_jd)
                        results.append((filename, fit_score))
                        print(f"Processed: {filename} | Fit Score: {fit_score:.3f}")
                    except:
                        results.append((filename, 0.0))
                else:
                    results.append((filename, 0.0))
        
        # Sort and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

if __name__ == "__main__":
    # Initialize screener
    screener = ResumeScreener()
    
    # Job description
    job_description = """
Dhulikhel Hospital, Kathmandu University Hospital, announces a vacancy for the post of Medical Officer for Dhulikhel Hospital and its outreach centers. Qualified and interested candidates are invited to submit their applications along with the required documents to the Communication Department by November 20, 2022.

Applicants must include the following documents with their application: an application letter, a valid NMC (Nepal Medical Council) registration certificate, a curriculum vitae (CV), a copy of their citizenship certificate, and academic-related documents. A few positions are available for the post of Medical Officer.
"""

    # Directory containing resumes
    resume_directory = "/home/manogyaguragai/Desktop/Resumes"
    
    # Get top matches
    top_matches = screener.screen_resumes(job_description, resume_directory, top_n=20)
    
    # Print results
    if top_matches:
        print("\nTOP MATCHES:")
        print("-" * 50)
        print(f"{'Filename':<40} | {'Fit Score':>9}")
        print("-" * 50)
        for filename, fit_score in top_matches:
            print(f"{filename[:39]:<40} | {fit_score:>9.3f}")
    else:
        print("No resumes processed successfully")
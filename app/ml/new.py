import os
import re
import PyPDF2
import time
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

class ResumeScreener:
    def __init__(self):
        # Use a small, efficient model for initial screening
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        return text
    
    def calculate_overall_similarity(self, resume_text, jd_text):
        """Calculate semantic similarity as overall score"""
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
            
        emb1 = self.model.encode(resume_text)
        emb2 = self.model.encode(jd_text)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def get_top_resumes(self, jd_text, pdf_directory, top_n=20):
        """Get top resumes based on overall similarity"""
        start_time = time.time()
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
                        overall_similarity = self.calculate_overall_similarity(clean_resume, clean_jd)
                        results.append({
                            "filename": filename,
                            "resume_text": resume_text,
                            "overall_similarity": overall_similarity
                        })
                        print(f"Processed: {filename} | Similarity: {overall_similarity:.3f}")
                    except:
                        results.append({
                            "filename": filename,
                            "resume_text": resume_text,
                            "overall_similarity": 0.0
                        })
                else:
                    results.append({
                        "filename": filename,
                        "resume_text": "",
                        "overall_similarity": 0.0
                    })
        
        # Sort and return top matches
        results.sort(key=lambda x: x["overall_similarity"], reverse=True)
        elapsed = time.time() - start_time
        print(f"\nInitial screening completed in {elapsed:.2f} seconds")
        return results[:top_n]
    
    def analyze_with_llm(self, jd_text, candidate):
        """Analyze candidate with LLM to compute detailed fit score"""
        SYSTEM_PROMPT = """
        You are an expert HR analyst. Your task is to analyze a candidate's resume against a job description and provide:
        1. An overall summary (1-2 sentences)
        2. A comprehensive fit score (0-100%) considering:
           - Technical skills (exact matches and transferable skills)
           - Non-technical skills
           - Experience relevance
           - Education qualifications
        3. Skill highlights (technical and non-technical)
        4. Experience highlights
        5. Education highlights
        6. Justification for the fit score
        
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
        - Be objective and critical in your assessment
        - Transferable skills should show how non-direct experience could be valuable
        - Fit score should be a percentage (0-100) reflecting overall suitability
        - Highlight most relevant qualifications
        """
        
        USER_PROMPT = f"""
        Job Description:
        {jd_text}
        
        Candidate Resume:
        {candidate['resume_text']}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return None

if __name__ == "__main__":
    # Initialize screener
    screener = ResumeScreener()
    
    # Job description
    job_description = """

About the job
Job Summary



We are looking for a highly skilled Artificial Intelligence (AI) Engineer to design, develop, and deploy AI-driven solutions that enhance our products and services. You will be responsible for building machine learning models, developing AI algorithms, and integrating AI capabilities into our existing systems. As an AI Engineer, you will work closely with data scientists, software engineers, and product teams to create intelligent, scalable, and high-performance AI applications.


Key Responsibilities



• Design, develop, and implement AI models, machine learning algorithms, and deep learning solutions.
• Train and fine-tune machine learning models using large datasets to improve accuracy and performance.
• Build and optimize AI-driven applications, including natural language processing (NLP), computer vision, recommendation systems, and predictive analytics.
• Develop and integrate AI models into cloud-based and edge computing environments.
• Work with structured and unstructured data, applying data preprocessing, feature engineering, and data augmentation techniques.
• Collaborate with cross-functional teams, including software engineers, data scientists, and business analysts, to align AI solutions with business objectives.
• Deploy AI models to production environments, monitor their performance, and continuously improve them based on real-world data.
• Research and stay up to date with the latest AI advancements, frameworks, and best practices to enhance our AI capabilities.
• Optimize AI models for efficiency, scalability, and real-time processing.
• Ensure AI models adhere to ethical AI principles, fairness, bias mitigation, and responsible AI deployment.


Required Skills & Qualifications



• Bachelor’s or Master’s degree in Computer Science, Artificial Intelligence, Data Science, or a related field.
• Proven experience in AI/ML model development, training, and deployment.
• Proficiency in programming languages such as Python, R, or Java.
• Hands-on experience with machine learning frameworks (TensorFlow, PyTorch, Scikit-learn, etc.).
• Strong understanding of deep learning, NLP, computer vision, reinforcement learning, and generative AI models.
• Experience working with cloud platforms (AWS, Azure, GCP) for AI model deployment.
• Knowledge of big data technologies such as Hadoop, Spark, or Apache Kafka is a plus.
• Experience with MLOps tools for model monitoring, retraining, and continuous deployment.
• Strong analytical and problem-solving skills with a keen interest in AI-driven solutions.
• Excellent communication and teamwork skills, with the ability to explain AI concepts to non-technical stakeholders.


Preferred Qualifications



• Experience in deploying AI models in real-world applications such as chatbots, fraud detection, predictive analytics, and robotics.
• Familiarity with AI ethics, model interpretability, and bias mitigation techniques.
• Knowledge of edge AI and IoT applications.
• Contributions to AI research or open-source AI projects.


If you are passionate about Artificial Intelligence Engineering and thrive in a fast-paced, innovative environment, we’d love to hear from you! 🚀

"""

    # Directory containing resumes
    resume_directory = "/home/manogyaguragai/Desktop/Resumes"
    
    # Get top 20 resumes based on overall similarity
    start_time = time.time()
    top_20 = screener.get_top_resumes(job_description, resume_directory, top_n=20)
    initial_time = time.time() - start_time
    
    # Analyze top 20 with LLM
    print("\nStarting detailed analysis with LLM...")
    llm_start_time = time.time()
    
    for candidate in top_20:
        analysis = screener.analyze_with_llm(job_description, candidate)
        if analysis:
            candidate.update(analysis)
            print(f"Analyzed: {candidate['filename']} | Fit Score: {analysis.get('fit_score', 0)}%")
        else:
            candidate["fit_score"] = 0
            print(f"Failed to analyze: {candidate['filename']}")
    
    llm_time = time.time() - llm_start_time
    
    # Sort by LLM fit score and get top 10
    top_20.sort(key=lambda x: x.get("fit_score", 0), reverse=True)
    final_top_10 = top_20[:10]
    
    # Print performance metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Initial screening time: {initial_time:.2f} seconds")
    print(f"LLM analysis time: {llm_time:.2f} seconds")
    print(f"Total processing time: {initial_time + llm_time:.2f} seconds")
    print(f"Resumes processed: {len(top_20)}")
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL TOP 10 CANDIDATES")
    print("="*80)
    print(f"{'Rank':<5} | {'Filename':<30} | {'Overall Similarity':>18} | {'Fit Score':>10}")
    print("-"*80)
    
    for i, candidate in enumerate(final_top_10, 1):
        print(f"{i:<5} | {candidate['filename'][:29]:<30} | "
              f"{candidate['overall_similarity']:>18.3f} | "
              f"{candidate.get('fit_score', 0):>10}%")
    
    # Print detailed reports
    print("\n" + "="*80)
    print("DETAILED CANDIDATE REPORTS")
    print("="*80)
    
    for i, candidate in enumerate(final_top_10, 1):
        print(f"\n{'='*40}")
        print(f"CANDIDATE #{i}: {candidate['filename']}")
        print(f"{'='*40}")
        print(f"Overall Summary: {candidate.get('overall_summary', 'N/A')}")
        print(f"Fit Score: {candidate.get('fit_score', 0)}%")
        
        print("\nTechnical Skills:")
        tech_skills = candidate.get('technical_skills', {})
        print(f"- Exact Matches: {', '.join(tech_skills.get('exact_matches', ['N/A']))}")
        print(f"- Transferable Skills: {', '.join(tech_skills.get('transferable_skills', ['N/A']))}")
        
        print(f"\nNon-technical Skills: {', '.join(candidate.get('non_technical_skills', ['N/A']))}")
        print(f"\nExperience Highlights: {candidate.get('experience_highlights', 'N/A')}")
        print(f"\nEducation Highlights: {candidate.get('education_highlights', 'N/A')}")
        print(f"\nJustification: {candidate.get('justification', 'N/A')}")
        print(f"{'='*40}\n")
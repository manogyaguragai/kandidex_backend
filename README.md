# Talynx: AI-Powered HR System Backend

An intelligent HR system backend built with FastAPI that leverages machine learning for resume ranking, candidate evaluation, and HR automation tasks.

## 🚀 Features

- **AI-Powered Resume Ranking**: Semantic similarity matching between job descriptions and resumes
- **Intelligent Question Generation**: Automated interview question generation based on resumes
- **Alternative Candidate Finding**: ML-driven candidate recommendation system
- **PDF Resume Processing**: Bulk resume parsing from PDFs and ZIP archives
- **Pre-trained BERT Model**: Fine-tuned on 10,000+ resumes for optimal matching
- **RESTful API**: Clean FastAPI endpoints with automatic documentation

## 🏗️ Architecture

The system uses a hybrid approach combining:
- **Bi-encoder** (BERT-mini): Pre-trained on 10,000+ resumes for fast semantic similarity
- **Cross-encoder** (MS-MARCO MiniLM): Precise relevance scoring and re-ranking
- **Ready-to-use Models**: No training required - optimized models included

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Git

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/manogyaguragai/talynx_backend.git
cd talynx-backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📦 Project Structure

```
talynx-backend/
│
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── routers/
│   │   ├── alternates.py       # Alternative candidates endpoint
│   │   ├── questions.py        # Question generation endpoint
│   │   └── ranking.py          # Resume ranking endpoint
│   ├── services/
│   │   ├── alternates_service.py
│   │   └── question_service.py
│   └── ml/
│       ├── bert.py            # BERT model training script
│       ├── kmeans.py          # Clustering algorithms
│       └── llm.py             # LLM integration
├── models/
│   └── output_bert_mini_job_resume/  # Pre-trained model (10K+ resumes)
├── data/
│   └── train.jsonl            # Training data (optional)
├── requirements.txt
└── README.md
```

## 🚀 Running the Application

### Standard Uvicorn

```bash
# Go to the app directory
cd app
```
And the run using Uvicorn

```bash
uvicorn app.main:app --reload
```

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# CORS Origins
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Model Paths
BERT_MODEL_PATH=./app/ml/output/output_bert_mini_job_resume/
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Device Configuration
DEVICE=cuda  # or cpu
```

## 🤖 Pre-trained Model

This system comes with a **pre-trained BERT-mini model** that has been fine-tuned on a dataset of **10,000+ resumes** for optimal job-resume matching performance. The model is already optimized and ready to use - no additional training required!

### Model Details
- **Base Model**: BERT-mini (lightweight and fast)
- **Training Dataset**: 10,000+ job-resume pairs
- **Fine-tuning**: Specialized for semantic matching between job descriptions and resumes
- **Performance**: Optimized for both accuracy and speed


## 📡 API Endpoints

### Resume Ranking
```http
POST /rank/
Content-Type: multipart/form-data

Parameters:
- job_desc: Job description text
- files: PDF files or ZIP archives containing resumes
```

### Question Generation
```http
GET /generate_questions/{resume_id}
```

### Alternative Candidates
```http
GET /find_alternatives/{employee_id}
```

### API Documentation

Once running, access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing the API

### Using cURL

```bash
# Test resume ranking
curl -X POST "http://localhost:8000/rank/" \
  -H "Content-Type: multipart/form-data" \
  -F "job_desc=Software Engineer with Python experience" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf"
```

### Using Python requests

```python
import requests

# Test endpoint
url = "http://localhost:8000/rank/"
files = [
    ('files', open('resume1.pdf', 'rb')),
    ('files', open('resume2.pdf', 'rb'))
]
data = {'job_desc': 'Software Engineer with Python experience'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## 🔧 Configuration

### CORS Configuration

Update `origins` in `main.py` to match your frontend URLs:

```python
origins = [
    "http://localhost:3000",    # React default
    "http://localhost:5173",    # Vite default
    "https://yourdomain.com",   # Production
]
```

### Model Configuration

To use different models, update the paths in `ranking.py`:

```python
# Custom trained model
bi_encoder = SentenceTransformer("path/to/your/model")

# Different cross-encoder
cross_encoder = CrossEncoder("your-preferred-cross-encoder")
```


## 📊 Performance Optimization

### GPU Acceleration

Ensure CUDA is properly installed:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Management

For large-scale processing:
```python
# Batch processing for multiple resumes
batch_size = 32
torch.cuda.empty_cache()  # Clear GPU memory
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**Model Loading Errors**
```bash
# Ensure model path exists
ls -la app/ml/output/output_bert_mini_job_resume/
```

**CORS Errors**
- Update origins in `main.py`
- Check frontend URL matches CORS configuration

**PDF Processing Issues**
```bash
# Install system dependencies
sudo apt-get install libmupdf-dev  # Ubuntu/Debian
brew install mupdf-tools           # macOS
```


**Manogya Guragai**
from fastapi import FastAPI
from routers import alternates, questions, ranking
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(title="Talynx: AI-Powered HR System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # allow specific origins
    allow_credentials=True,
    allow_methods=["*"],              # allow all HTTP methods
    allow_headers=["*"],              # allow all headers
)

app.include_router(alternates.router)
app.include_router(questions.router)
app.include_router(ranking.router)

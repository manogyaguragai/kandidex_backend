from fastapi import FastAPI
from routers import alternates, questions, ranking, screening_runs
from auth import auth as auth_router
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(title="Kandidex: AI-Powered HR System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alternates.router)
app.include_router(questions.router)
app.include_router(ranking.router)
app.include_router(auth_router.router)
app.include_router(screening_runs.router)

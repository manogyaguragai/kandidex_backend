from pydantic import BaseModel
from typing import List

class Employee(BaseModel):
    id: str
    name: str
    resume_vector: List[float]
    cluster_id: int
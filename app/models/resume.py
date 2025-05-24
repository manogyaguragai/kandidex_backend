from pydantic import BaseModel
from typing import List, Optional

class Resume(BaseModel):
    id: Optional[str]
    name: str
    content: str

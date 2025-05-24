from fastapi import APIRouter
from services.alternates_service import find_alternatives

router = APIRouter(prefix="/find_alternatives", tags=["Alternatives"])

@router.get("/{employee_id}")
async def get_alternatives(employee_id: str):
    return await find_alternatives(employee_id)
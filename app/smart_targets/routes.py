from fastapi import APIRouter

from app.smart_targets.service import SmartTargetsService

router = APIRouter()


@router.get(path="/smart")
async def smart():
    await SmartTargetsService().ask_question()

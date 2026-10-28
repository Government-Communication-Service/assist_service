from datetime import datetime

from fastapi import APIRouter, status

router = APIRouter()

start_time = datetime.now()
IS_READY = True


@router.get("", status_code=status.HTTP_200_OK)
def get_health_check():
    return {"status": "fine"}

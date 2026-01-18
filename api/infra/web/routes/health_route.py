"""
Health check routes
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "message": "Server is running",
            "timestamp": datetime.now().isoformat(),
        },
        status_code=200,
    )

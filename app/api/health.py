"""
Health check API endpoints
시스템 상태 확인 엔드포인트
"""
import os
import time
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..lib.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 시작 시간 기록
start_time = time.time()

class HealthResponse(BaseModel):
    """Health 체크 응답 모델"""
    status: str
    timestamp: str
    uptime: float
    version: str = "2.0.0"
    environment: str
    
class StatsResponse(BaseModel):
    """시스템 통계 응답 모델"""
    uptime: float
    uptime_human: str
    cpu_percent: float
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]
    system_info: Dict[str, Any]

def get_uptime() -> float:
    """업타임 반환 (초)"""
    return time.time() - start_time

def format_uptime(seconds: float) -> str:
    """업타임 포맷팅"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def get_memory_info() -> Dict[str, Any]:
    """메모리 사용량 정보"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percentage": memory.percent,
        "total_gb": round(memory.total / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2)
    }

def get_disk_info() -> Dict[str, Any]:
    """디스크 사용량 정보"""
    disk = psutil.disk_usage('/')
    return {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
        "percentage": round((disk.used / disk.total) * 100, 2),
        "total_gb": round(disk.total / (1024**3), 2),
        "used_gb": round(disk.used / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2)
    }

def get_system_info() -> Dict[str, Any]:
    """시스템 정보"""
    return {
        "platform": os.name,
        "python_version": os.sys.version.split()[0],
        "cpu_count": psutil.cpu_count(),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """기본 헬스 체크"""
    try:
        return HealthResponse(
            status="OK",
            timestamp=datetime.now().isoformat(),
            uptime=get_uptime(),
            environment=os.getenv("NODE_ENV", "development")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="ERROR",
            timestamp=datetime.now().isoformat(),
            uptime=get_uptime(),
            environment=os.getenv("NODE_ENV", "development")
        )

@router.get("/stats", response_model=StatsResponse)
async def system_stats():
    """시스템 통계"""
    try:
        uptime = get_uptime()
        
        return StatsResponse(
            uptime=uptime,
            uptime_human=format_uptime(uptime),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_usage=get_memory_info(),
            disk_usage=get_disk_info(),
            system_info=get_system_info()
        )
    except Exception as e:
        logger.error(f"Stats collection failed: {e}")
        raise

@router.get("/ping")
async def ping():
    """간단한 ping 엔드포인트"""
    return {"message": "pong", "timestamp": datetime.now().isoformat()}
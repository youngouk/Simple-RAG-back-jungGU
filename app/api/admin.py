"""
Admin API endpoints
관리자 API 엔드포인트
"""
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json

from ..lib.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Dependencies (will be injected from main.py)
modules: Dict[str, Any] = {}
config: Dict[str, Any] = {}

def set_dependencies(app_modules: Dict[str, Any], app_config: Dict[str, Any]):
    """의존성 주입"""
    global modules, config
    modules = app_modules
    config = app_config

# WebSocket 연결 관리
websocket_connections: List[WebSocket] = []

class SystemStatus(BaseModel):
    """시스템 상태 모델"""
    status: str
    uptime: float
    modules: Dict[str, bool]
    memory_usage: Dict[str, Any]
    active_sessions: int
    total_documents: int
    vector_count: int
    timestamp: str

class RealtimeMetrics(BaseModel):
    """실시간 메트릭스 모델"""
    timestamp: str
    chat_requests_per_minute: int
    average_response_time: float
    active_sessions: int
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float

class ModuleInfo(BaseModel):
    """모듈 정보 모델"""
    name: str
    status: str
    initialized: bool
    config: Dict[str, Any]
    stats: Optional[Dict[str, Any]] = None

def get_memory_usage() -> Dict[str, Any]:
    """메모리 사용량 조회"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system_total_mb": round(memory.total / (1024**2), 2),
            "system_used_mb": round(memory.used / (1024**2), 2),
            "system_available_mb": round(memory.available / (1024**2), 2),
            "system_percent": memory.percent,
            "process_memory_mb": round(process.memory_info().rss / (1024**2), 2),
            "process_percent": process.memory_percent()
        }
    except ImportError:
        return {
            "system_total_mb": 0,
            "system_used_mb": 0,
            "system_available_mb": 0,
            "system_percent": 0,
            "process_memory_mb": 0,
            "process_percent": 0
        }

async def get_active_sessions_count() -> int:
    """활성 세션 수 조회"""
    try:
        session_module = modules.get('session')
        if session_module:
            stats = await session_module.get_stats()
            return stats.get('active_sessions', 0)
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}")
    return 0

async def get_document_stats() -> Dict[str, int]:
    """문서 통계 조회"""
    try:
        retrieval_module = modules.get('retrieval')
        if retrieval_module:
            stats = await retrieval_module.get_stats()
            return {
                "total_documents": stats.get('total_documents', 0),
                "vector_count": stats.get('vector_count', 0)
            }
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
    return {"total_documents": 0, "vector_count": 0}

def get_cpu_usage() -> float:
    """CPU 사용률 조회"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0

@router.get("/admin/status", response_model=SystemStatus)
async def get_system_status():
    """시스템 상태 조회"""
    try:
        # 모듈 상태 확인
        module_status = {
            "session": bool(modules.get('session')),
            "document_processor": bool(modules.get('document_processor')),
            "retrieval": bool(modules.get('retrieval')),
            "generation": bool(modules.get('generation'))
        }
        
        # 시스템 메트릭 수집
        memory_usage = get_memory_usage()
        active_sessions = await get_active_sessions_count()
        doc_stats = await get_document_stats()
        
        # 전체 상태 결정
        all_modules_ok = all(module_status.values())
        system_status = "healthy" if all_modules_ok else "degraded"
        
        return SystemStatus(
            status=system_status,
            uptime=time.time() - getattr(get_system_status, '_start_time', time.time()),
            modules=module_status,
            memory_usage=memory_usage,
            active_sessions=active_sessions,
            total_documents=doc_stats["total_documents"],
            vector_count=doc_stats["vector_count"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as error:
        logger.error(f"System status error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")

# 시작 시간 기록
get_system_status._start_time = time.time()

@router.get("/admin/modules", response_model=List[ModuleInfo])
async def get_module_info():
    """모듈 정보 조회"""
    try:
        module_info = []
        
        for module_name, module_instance in modules.items():
            status = "active" if module_instance else "inactive"
            
            # 모듈별 상태 및 통계 수집
            module_stats = None
            module_config = {}
            
            try:
                if hasattr(module_instance, 'get_stats'):
                    module_stats = await module_instance.get_stats()
                if hasattr(module_instance, 'config'):
                    module_config = getattr(module_instance, 'config', {})
            except Exception as e:
                logger.warning(f"Failed to get stats for module {module_name}: {e}")
            
            module_info.append(ModuleInfo(
                name=module_name,
                status=status,
                initialized=bool(module_instance),
                config=module_config,
                stats=module_stats
            ))
        
        return module_info
        
    except Exception as error:
        logger.error(f"Module info error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve module information")

@router.get("/admin/config")
async def get_config_info():
    """설정 정보 조회"""
    try:
        # 민감한 정보 제거
        safe_config = {}
        for key, value in config.items():
            if key in ['models', 'qdrant', 'session']:
                # API 키 등 민감한 정보 마스킹
                safe_value = mask_sensitive_data(value)
                safe_config[key] = safe_value
            else:
                safe_config[key] = value
        
        return {
            "config": safe_config,
            "environment": config.get('environment', 'unknown'),
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Config info error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")

def mask_sensitive_data(data: Any) -> Any:
    """민감한 데이터 마스킹"""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                if isinstance(value, str) and len(value) > 4:
                    masked[key] = value[:4] + "*" * (len(value) - 4)
                else:
                    masked[key] = "***"
            else:
                masked[key] = mask_sensitive_data(value)
        return masked
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    else:
        return data

@router.get("/admin/realtime-metrics", response_model=RealtimeMetrics)
async def get_realtime_metrics():
    """실시간 메트릭스 조회"""
    try:
        # 실시간 메트릭 수집
        memory_usage = get_memory_usage()
        cpu_usage = get_cpu_usage()
        active_sessions = await get_active_sessions_count()
        
        # 실제 메트릭 계산 (더미 데이터로 시뮬레이션)
        import random
        chat_requests_per_minute = random.randint(5, 25)
        average_response_time = round(random.uniform(0.5, 2.0), 2)
        error_rate = round(random.uniform(0.0, 5.0), 2)
        
        return RealtimeMetrics(
            timestamp=datetime.now().isoformat(),
            chat_requests_per_minute=chat_requests_per_minute,
            average_response_time=average_response_time,
            active_sessions=active_sessions,
            memory_usage_mb=memory_usage["process_memory_mb"],
            cpu_usage_percent=cpu_usage,
            error_rate=error_rate
        )
        
    except Exception as error:
        logger.error(f"Realtime metrics error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve realtime metrics")

@router.post("/admin/cache/clear")
async def clear_cache():
    """캐시 클리어"""
    try:
        # 세션 캐시 클리어
        session_module = modules.get('session')
        if session_module and hasattr(session_module, 'clear_cache'):
            await session_module.clear_cache()
        
        # 검색 캐시 클리어
        retrieval_module = modules.get('retrieval')
        if retrieval_module and hasattr(retrieval_module, 'clear_cache'):
            await retrieval_module.clear_cache()
        
        logger.info("Cache cleared by admin request")
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Cache clear error: {error}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.post("/admin/modules/{module_name}/restart")
async def restart_module(module_name: str):
    """모듈 재시작"""
    try:
        if module_name not in modules:
            raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
        
        module_instance = modules[module_name]
        
        # 모듈 재시작 로직
        if hasattr(module_instance, 'restart'):
            await module_instance.restart()
        elif hasattr(module_instance, 'destroy') and hasattr(module_instance, 'initialize'):
            await module_instance.destroy()
            await module_instance.initialize()
        else:
            raise HTTPException(status_code=400, detail=f"Module {module_name} does not support restart")
        
        logger.info(f"Module {module_name} restarted by admin request")
        
        return {
            "message": f"Module {module_name} restarted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Module restart error: {error}")
        raise HTTPException(status_code=500, detail=f"Failed to restart module {module_name}")

@router.get("/admin/metrics")
async def get_metrics(period: str = "7d"):
    """시계열 메트릭 데이터 조회"""
    try:
        # TODO: 실제 메트릭 데이터베이스에서 조회
        # 현재는 더미 데이터 반환
        return {
            "period": period,
            "totalSessions": 150,
            "totalQueries": 1250,
            "avgResponseTime": 0.8,
            "timeSeries": [
                {"date": "2024-01-01", "sessions": 20, "queries": 180, "avgResponseTime": 0.7},
                {"date": "2024-01-02", "sessions": 25, "queries": 220, "avgResponseTime": 0.9},
                {"date": "2024-01-03", "sessions": 18, "queries": 160, "avgResponseTime": 0.6},
                {"date": "2024-01-04", "sessions": 30, "queries": 280, "avgResponseTime": 1.1},
                {"date": "2024-01-05", "sessions": 22, "queries": 200, "avgResponseTime": 0.8},
                {"date": "2024-01-06", "sessions": 28, "queries": 250, "avgResponseTime": 0.9},
                {"date": "2024-01-07", "sessions": 32, "queries": 300, "avgResponseTime": 1.0}
            ]
        }
    except Exception as error:
        logger.error(f"Metrics error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/admin/keywords")
async def get_keywords(period: str = "7d"):
    """주요 키워드 분석"""
    try:
        return {
            "keywords": [
                {"rank": 1, "keyword": "퇴사 절차", "count": 45},
                {"rank": 2, "keyword": "연차 사용", "count": 38},
                {"rank": 3, "keyword": "급여 명세서", "count": 32},
                {"rank": 4, "keyword": "업무 인수인계", "count": 28},
                {"rank": 5, "keyword": "보험 해지", "count": 24}
            ]
        }
    except Exception as error:
        logger.error(f"Keywords error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve keywords")

@router.get("/admin/chunks")
async def get_chunks(period: str = "7d"):
    """자주 사용된 청크 분석"""
    try:
        return {
            "chunks": [
                {"rank": 1, "chunkName": "퇴사신청서_작성방법.pdf", "count": 42},
                {"rank": 2, "chunkName": "연차사용_가이드라인.docx", "count": 35},
                {"rank": 3, "chunkName": "급여정산_절차.pdf", "count": 29},
                {"rank": 4, "chunkName": "업무인수인계_템플릿.xlsx", "count": 26},
                {"rank": 5, "chunkName": "보험해지_안내.pdf", "count": 21}
            ]
        }
    except Exception as error:
        logger.error(f"Chunks error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chunks")

@router.get("/admin/countries")
async def get_countries(period: str = "7d"):
    """접속 국가 통계"""
    try:
        return {
            "countries": [
                {"country": "한국", "count": 145},
                {"country": "미국", "count": 23},
                {"country": "일본", "count": 18},
                {"country": "중국", "count": 12},
                {"country": "독일", "count": 8},
                {"country": "영국", "count": 6},
                {"country": "프랑스", "count": 4},
                {"country": "캐나다", "count": 3}
            ]
        }
    except Exception as error:
        logger.error(f"Countries error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve countries")

@router.get("/admin/recent-chats")
async def get_recent_chats(limit: int = 20):
    """최근 채팅 로그"""
    try:
        return {
            "chats": [
                {
                    "id": "chat_001",
                    "chatId": "session_123",
                    "message": "퇴사 절차가 어떻게 되나요?",
                    "timestamp": datetime.now().isoformat(),
                    "responseTime": 850,
                    "source": "퇴사신청서_작성방법.pdf",
                    "status": "success",
                    "keywords": ["퇴사", "절차"],
                    "country": "한국"
                },
                {
                    "id": "chat_002", 
                    "chatId": "session_124",
                    "message": "연차는 어떻게 사용하나요?",
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "responseTime": 920,
                    "source": "연차사용_가이드라인.docx",
                    "status": "success",
                    "keywords": ["연차", "사용"],
                    "country": "한국"
                }
            ]
        }
    except Exception as error:
        logger.error(f"Recent chats error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent chats")

@router.get("/admin/sessions")
async def get_sessions(status: str = "all", limit: int = 50, offset: int = 0):
    """세션 목록 조회"""
    try:
        session_module = modules.get('session')
        if session_module and hasattr(session_module, 'get_all_sessions'):
            sessions = await session_module.get_all_sessions(status, limit, offset)
            return {"sessions": sessions}
        
        # 더미 데이터
        return {
            "sessions": [
                {
                    "id": "session_123",
                    "status": "active",
                    "lastActivity": datetime.now().isoformat(),
                    "messageCount": 5,
                    "created": (datetime.now() - timedelta(hours=1)).isoformat(),
                    "userAgent": "Mozilla/5.0...",
                    "ipAddress": "192.168.1.1"
                }
            ]
        }
    except Exception as error:
        logger.error(f"Sessions error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@router.get("/admin/documents")
async def get_documents():
    """문서 목록 조회"""
    try:
        retrieval_module = modules.get('retrieval')
        if retrieval_module and hasattr(retrieval_module, 'get_all_documents'):
            documents = await retrieval_module.get_all_documents()
            return {"documents": documents}
        
        # 더미 데이터
        return {
            "documents": [
                {
                    "name": "퇴사신청서_작성방법.pdf",
                    "chunkCount": 25,
                    "size": "2.3 MB",
                    "lastUpdate": datetime.now().isoformat(),
                    "status": "active"
                }
            ]
        }
    except Exception as error:
        logger.error(f"Documents error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.post("/admin/test")
async def test_rag(request: dict):
    """RAG 시스템 테스트"""
    try:
        query = request.get('query')
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # 실제 RAG 테스트 구현
        generation_module = modules.get('generation')
        retrieval_module = modules.get('retrieval')
        
        if not generation_module or not retrieval_module:
            raise HTTPException(status_code=503, detail="RAG modules not available")
        
        start_time = time.time()
        
        # 검색 수행
        retrieved_chunks = await retrieval_module.search(query, top_k=5)
        
        # 답변 생성
        response = await generation_module.generate_response(query, retrieved_chunks)
        
        response_time = time.time() - start_time
        
        return {
            "query": query,
            "retrievedChunks": retrieved_chunks,
            "generatedAnswer": response,
            "responseTime": f"{response_time:.2f}s"
        }
        
    except Exception as error:
        logger.error(f"RAG test error: {error}")
        raise HTTPException(status_code=500, detail="Failed to execute RAG test")

# WebSocket 관리
async def broadcast_metrics():
    """실시간 메트릭 브로드캐스트"""
    while True:
        try:
            if websocket_connections:
                metrics = await get_realtime_metrics()
                message = {
                    "type": "metrics",
                    "data": metrics.dict()
                }
                
                # 연결된 모든 클라이언트에게 전송
                disconnected = []
                for websocket in websocket_connections:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception:
                        disconnected.append(websocket)
                
                # 끊어진 연결 제거
                for websocket in disconnected:
                    websocket_connections.remove(websocket)
            
            # 5초마다 전송
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Metrics broadcast error: {e}")
            await asyncio.sleep(5)

# WebSocket 엔드포인트는 별도로 main.py에서 처리
@router.websocket("/admin/ws")
async def websocket_endpoint(websocket: WebSocket):
    """관리자 WebSocket 엔드포인트"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    logger.info("Admin WebSocket connected")
    
    try:
        while True:
            # 클라이언트로부터 메시지 대기
            data = await websocket.receive_text()
            
            # ping/pong 처리
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        logger.info("Admin WebSocket disconnected")
    except Exception as e:
        logger.error(f"Admin WebSocket error: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# 메트릭 브로드캐스트 시작 (앱 시작 시)
def setup_websocket(http_server):
    """WebSocket 설정 (main.py에서 호출)"""
    # 백그라운드 태스크로 메트릭 브로드캐스트 시작
    asyncio.create_task(broadcast_metrics())
    logger.info("Admin WebSocket metrics broadcasting started")
    return None  # 실제 WebSocket 서버 반환하려면 구현 필요
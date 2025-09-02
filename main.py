"""
RAG Chatbot FastAPI Application
한국어 RAG 챗봇 시스템의 메인 애플리케이션
"""
import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add app directory to path
app_dir = Path(__file__).parent / "app"
sys.path.append(str(app_dir))

from app.lib.config_loader import ConfigLoader
from app.lib.logger import get_logger
from app.api import chat, upload, admin, health
from app.modules.enhanced_session import EnhancedSessionModule
from app.modules.document_processing import DocumentProcessor
from app.modules.retrieval_rerank import RetrievalModule
from app.modules.generation import GenerationModule

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

class RAGChatbotApp:
    """RAG 챗봇 메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.config = None
        self.modules = {}
        self.app = None
        
    async def initialize_modules(self):
        """모듈 초기화"""
        try:
            logger.info("🔧 Initializing modules...")
            
            # 1. 설정 로드
            config_loader = ConfigLoader()
            self.config = config_loader.load_config()
            
            # 2. 세션 모듈
            logger.info("Initializing session module...")
            self.modules['session'] = EnhancedSessionModule(self.config)
            await self.modules['session'].initialize()
            
            # 3. 문서 처리 모듈
            logger.info("Initializing document processing module...")
            self.modules['document_processor'] = DocumentProcessor(self.config)
            
            # 4. 검색 및 리랭킹 모듈
            logger.info("Initializing retrieval module...")
            self.modules['retrieval'] = RetrievalModule(
                self.config, 
                self.modules['document_processor'].embedder
            )
            await self.modules['retrieval'].initialize()
            
            # 5. 답변 생성 모듈
            logger.info("Initializing generation module...")
            self.modules['generation'] = GenerationModule(self.config)
            await self.modules['generation'].initialize()
            
            logger.info("✅ All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Module initialization failed: {e}")
            raise
    
    async def cleanup_modules(self):
        """모듈 정리"""
        try:
            logger.info("🧹 Cleaning up modules...")
            
            if 'session' in self.modules:
                await self.modules['session'].destroy()
                
            if 'retrieval' in self.modules:
                await self.modules['retrieval'].close()
                
            if 'generation' in self.modules:
                await self.modules['generation'].destroy()
                
            logger.info("✅ Module cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# 글로벌 앱 인스턴스
rag_app = RAGChatbotApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    try:
        logger.info("🚀 Starting RAG Chatbot Application...")
        await rag_app.initialize_modules()
        
        # 라우터에 의존성 주입
        chat.set_dependencies(rag_app.modules, rag_app.config)
        upload.set_dependencies(rag_app.modules, rag_app.config)
        admin.set_dependencies(rag_app.modules, rag_app.config)
        
        logger.info("✅ Application started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
        
    yield
    
    # 종료 시
    await rag_app.cleanup_modules()
    logger.info("📡 Application shutdown completed")

# FastAPI 앱 생성
app = FastAPI(
    title="RAG Chatbot API",
    description="경량 모듈형 한국어 RAG 챗봇 시스템",
    version="2.0.0",
    lifespan=lifespan
)

# Rate limiting 설정
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 미들웨어 설정
# 배포 환경에서의 CORS 허용 도메인은 환경 변수 ALLOWED_ORIGINS(콤마 구분)로 확장 가능
default_allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
]
env_allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_allowed_origins:
    default_allowed_origins.extend([
        origin.strip() for origin in env_allowed_origins.split(",") if origin.strip()
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙 (있는 경우)
static_path = Path(__file__).parent.parent / "public"
if static_path.exists():
    app.mount("/dashboard", StaticFiles(directory=static_path), name="static")

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(admin.router, prefix="/api", tags=["Admin"])

@app.get("/")
async def root():
    """루트 엔드포인트 - 대시보드로 리다이렉트"""
    return RedirectResponse(url="/dashboard/admin.html")

@app.get("/api")
async def api_info():
    """API 정보 엔드포인트"""
    return {
        "name": "RAG Chatbot API",
        "version": "2.0.0",
        "description": "경량 모듈형 한국어 RAG 챗봇 시스템",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "dashboard": "/dashboard",
            "chat": "/api/chat",
            "upload": "/api/upload",
            "documents": "/api/upload/documents",
            "admin": "/api/admin"
        }
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = asyncio.get_event_loop().time()
    
    response = await call_next(request)
    
    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
        extra={
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request.client.host if request.client else None
        }
    )
    
    return response

def main():
    """메인 실행 함수"""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )

if __name__ == "__main__":
    main()
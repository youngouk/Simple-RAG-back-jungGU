"""
Structured logging for RAG Chatbot
구조화된 로깅 시스템
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from structlog.stdlib import LoggerFactory


class RAGLogger:
    """RAG 챗봇 로깅 시스템"""
    
    def __init__(self):
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_dir = Path(__file__).parent.parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """로깅 설정"""
        # 로그 레벨 설정
        level = getattr(logging, self.log_level, logging.INFO)
        
        # 기본 로거 설정
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_dir / "app.log")
            ]
        )
        
        # Structlog 설정
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_context,
                structlog.processors.JSONRenderer() if self._should_use_json() else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _should_use_json(self) -> bool:
        """JSON 형식 사용 여부 결정"""
        return os.getenv("LOG_FORMAT", "console").lower() == "json"
    
    def _add_context(self, logger, method_name, event_dict):
        """컨텍스트 정보 추가"""
        event_dict["service"] = "rag-chatbot"
        event_dict["environment"] = os.getenv("NODE_ENV", "development")
        event_dict["pid"] = os.getpid()
        return event_dict
    
    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """구조화된 로거 반환"""
        return structlog.get_logger(name or __name__)


# 글로벌 로거 인스턴스
_rag_logger = RAGLogger()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """로거 인스턴스 반환"""
    return _rag_logger.get_logger(name)


class ChatLoggingMiddleware:
    """채팅 요청 로깅 미들웨어"""
    
    def __init__(self):
        self.logger = get_logger("chat_middleware")
    
    async def log_chat_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any], 
                             processing_time: float, session_id: str = None):
        """채팅 요청/응답 로깅"""
        log_data = {
            "event": "chat_request",
            "session_id": session_id,
            "message_length": len(request_data.get("message", "")),
            "response_length": len(response_data.get("answer", "")),
            "processing_time": processing_time,
            "tokens_used": response_data.get("tokens_used", 0),
            "sources_count": len(response_data.get("sources", [])),
            "success": "error" not in response_data
        }
        
        if response_data.get("error"):
            self.logger.error("Chat request failed", **log_data, error=response_data["error"])
        else:
            self.logger.info("Chat request completed", **log_data)


def create_chat_logging_middleware():
    """채팅 로깅 미들웨어 팩토리"""
    return ChatLoggingMiddleware()
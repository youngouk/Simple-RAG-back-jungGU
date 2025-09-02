"""
Test Configuration and Fixtures
테스트 설정 및 픽스처 정의
"""
import os
import tempfile
import asyncio
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

# 테스트 환경 변수 설정
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"

# Import after setting environment
from app.lib.config_loader import ConfigLoader
from app.modules.session import SessionModule
from app.modules.document_processing import DocumentProcessor
from app.modules.retrieval_rerank import RetrievalModule
from app.modules.generation import GenerationModule
from main import app, rag_app


@pytest.fixture(scope="session")
def event_loop():
    """이벤트 루프 픽스처"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """임시 디렉토리 픽스처"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> Dict[str, Any]:
    """테스트용 설정 픽스처"""
    return {
        "app": {
            "name": "RAG Chatbot Test",
            "version": "2.0.0-test",
            "debug": True
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "collection_name": "test_documents",
            "vector_size": 768,
            "distance": "cosine"
        },
        "embeddings": {
            "provider": "google",
            "model": "text-embedding-004",
            "batch_size": 100
        },
        "llm": {
            "providers": ["google"],
            "google": {
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.3,
                "max_tokens": 2048
            }
        },
        "document_processing": {
            "splitter_type": "recursive",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "max_file_size": 50 * 1024 * 1024  # 50MB
        },
        "session": {
            "ttl": 3600,
            "max_exchanges": 5
        },
        "logging": {
            "level": "DEBUG",
            "format": "structured"
        },
        "uploads": {
            "directory": str(temp_dir / "uploads"),
            "max_file_size": 50 * 1024 * 1024,
            "allowed_types": [".pdf", ".txt", ".md", ".docx", ".xlsx"]
        }
    }


@pytest.fixture
def mock_config_loader(test_config):
    """Mock 설정 로더 픽스처"""
    mock = Mock(spec=ConfigLoader)
    mock.load_config.return_value = test_config
    return mock


@pytest_asyncio.fixture
async def mock_session_module(test_config):
    """Mock 세션 모듈 픽스처"""
    mock = AsyncMock(spec=SessionModule)
    mock.initialize.return_value = None
    mock.destroy.return_value = None
    mock.get_session.return_value = {"session_id": "test-session", "exchanges": []}
    mock.add_exchange.return_value = None
    return mock


@pytest_asyncio.fixture
async def mock_document_processor(test_config):
    """Mock 문서 처리 모듈 픽스처"""
    mock = Mock(spec=DocumentProcessor)
    mock.supported_loaders = {".pdf": "pdf", ".txt": "text", ".md": "markdown"}
    mock.embedder = Mock()
    mock.embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
    mock.embedder.embed_documents = AsyncMock(return_value=[[0.1] * 768])
    
    mock.process_document = AsyncMock(return_value={
        "document_id": "test-doc-123",
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "content": "테스트 문서 내용입니다.",
                "metadata": {"page": 1}
            }
        ],
        "processing_time": 1.5,
        "loader_type": "pdf",
        "splitter_type": "recursive"
    })
    
    return mock


@pytest_asyncio.fixture
async def mock_retrieval_module(test_config):
    """Mock 검색 모듈 픽스처"""
    mock = AsyncMock(spec=RetrievalModule)
    mock.initialize.return_value = None
    mock.close.return_value = None
    
    mock.search = AsyncMock(return_value=[
        {
            "chunk_id": "chunk-1",
            "content": "테스트 문서 내용입니다.",
            "score": 0.95,
            "metadata": {"document_id": "test-doc-123", "page": 1}
        }
    ])
    
    return mock


@pytest_asyncio.fixture
async def mock_generation_module(test_config):
    """Mock 생성 모듈 픽스처"""
    mock = AsyncMock(spec=GenerationModule)
    mock.initialize.return_value = None
    mock.destroy.return_value = None
    
    mock.generate_response = AsyncMock(return_value={
        "response": "테스트 응답입니다.",
        "model": "gemini-2.0-flash-exp",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    })
    
    return mock


@pytest_asyncio.fixture
async def mock_modules(
    mock_session_module,
    mock_document_processor,
    mock_retrieval_module,
    mock_generation_module
):
    """모든 Mock 모듈을 포함하는 픽스처"""
    return {
        "session": mock_session_module,
        "document_processor": mock_document_processor,
        "retrieval": mock_retrieval_module,
        "generation": mock_generation_module
    }


@pytest.fixture
def test_client(mock_modules, test_config):
    """테스트 클라이언트 픽스처"""
    # Mock dependencies injection
    from app.api import chat, upload, admin
    
    chat.set_dependencies(mock_modules, test_config)
    upload.set_dependencies(mock_modules, test_config)
    admin.set_dependencies(mock_modules, test_config)
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_pdf_file(temp_dir: Path):
    """샘플 PDF 파일 픽스처"""
    pdf_path = temp_dir / "sample.pdf"
    # Create a minimal PDF-like file for testing
    pdf_path.write_bytes(b"%PDF-1.4\n%Test PDF content\n%%EOF")
    return pdf_path


@pytest.fixture
def sample_text_file(temp_dir: Path):
    """샘플 텍스트 파일 픽스처"""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text("이것은 테스트용 문서입니다.\n한국어 텍스트가 포함되어 있습니다.", encoding="utf-8")
    return txt_path


@pytest.fixture
def large_file(temp_dir: Path):
    """큰 파일 픽스처 (업로드 제한 테스트용)"""
    large_path = temp_dir / "large.txt"
    # Create a file larger than 50MB
    content = "A" * (51 * 1024 * 1024)  # 51MB
    large_path.write_text(content)
    return large_path


# Test data fixtures
@pytest.fixture
def sample_chat_message():
    """샘플 채팅 메시지 픽스처"""
    return {
        "message": "RAG 챗봇에 대해 설명해주세요.",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_document_metadata():
    """샘플 문서 메타데이터 픽스처"""
    return {
        "document_id": "test-doc-123",
        "filename": "sample.pdf",
        "file_size": 1024,
        "chunks": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "first_chunk_content": "이것은 테스트 문서의 첫 번째 청크입니다."
    }


# Mock environment fixtures
@pytest.fixture
def mock_env_vars(monkeypatch):
    """환경 변수 Mock 픽스처"""
    test_vars = {
        "GOOGLE_API_KEY": "test-google-api-key",
        "ANTHROPIC_API_KEY": "test-anthropic-api-key",
        "OPENAI_API_KEY": "test-openai-api-key",
        "QDRANT_URL": "http://localhost:6333",
        "DEBUG": "true",
        "ENVIRONMENT": "test"
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars


# Performance test fixtures
@pytest.fixture
def performance_timer():
    """성능 측정 타이머 픽스처"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()
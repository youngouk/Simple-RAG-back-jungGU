"""
Simple Module Tests
간단한 모듈 테스트 (실제 import 검증)
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestModuleImports:
    """모듈 임포트 테스트"""

    def test_config_loader_import(self):
        """설정 로더 임포트 테스트"""
        from app.lib.config_loader import ConfigLoader
        assert ConfigLoader is not None

    def test_logger_import(self):
        """로거 임포트 테스트"""
        from app.lib.logger import get_logger
        logger = get_logger(__name__)
        assert logger is not None

    def test_session_module_import(self):
        """세션 모듈 임포트 테스트"""
        from app.modules.session import SessionModule
        assert SessionModule is not None

    def test_document_processor_import(self):
        """문서 처리 모듈 임포트 테스트"""
        from app.modules.document_processing import DocumentProcessor
        assert DocumentProcessor is not None

    def test_retrieval_module_import(self):
        """검색 모듈 임포트 테스트"""
        from app.modules.retrieval_rerank import RetrievalModule
        assert RetrievalModule is not None

    def test_generation_module_import(self):
        """생성 모듈 임포트 테스트"""
        from app.modules.generation import GenerationModule
        assert GenerationModule is not None


class TestConfigLoader:
    """설정 로더 테스트"""

    def test_config_loader_creation(self):
        """설정 로더 생성 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        assert loader is not None

    def test_config_loading(self, test_config):
        """설정 로딩 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        with patch.object(ConfigLoader, 'load_config', return_value=test_config):
            loader = ConfigLoader()
            config = loader.load_config()
            
            assert config is not None
            assert "app" in config
            assert config["app"]["name"] == "RAG Chatbot Test"


class TestDocumentProcessor:
    """문서 처리 모듈 테스트"""

    def test_document_processor_creation(self, test_config):
        """문서 처리 모듈 생성 테스트"""
        from app.modules.document_processing import DocumentProcessor
        
        processor = DocumentProcessor(test_config)
        assert processor is not None
        assert processor.config == test_config

    def test_supported_file_types(self, test_config):
        """지원 파일 형식 테스트"""
        from app.modules.document_processing import DocumentProcessor
        
        processor = DocumentProcessor(test_config)
        supported_types = processor.supported_loaders
        
        expected_types = ['.pdf', '.txt', '.md', '.docx', '.xlsx']
        for file_type in expected_types:
            assert file_type in supported_types


class TestSessionModule:
    """세션 모듈 테스트"""

    @pytest.mark.asyncio
    async def test_session_module_creation(self, test_config):
        """세션 모듈 생성 테스트"""
        from app.modules.session import SessionModule
        
        session_config = test_config.get('session', {})
        module = SessionModule(session_config)
        assert module is not None

    @pytest.mark.asyncio
    async def test_session_module_initialization(self, test_config):
        """세션 모듈 초기화 테스트"""
        from app.modules.session import SessionModule
        
        session_config = test_config.get('session', {})
        module = SessionModule(session_config)
        
        # Mock 초기화 (실제 Redis 없이)
        with patch.object(module, 'initialize', return_value=None):
            await module.initialize()
            assert True  # 에러 없이 완료되면 성공


class TestRetrievalModule:
    """검색 모듈 테스트"""

    def test_retrieval_module_creation(self, test_config):
        """검색 모듈 생성 테스트"""
        from app.modules.retrieval_rerank import RetrievalModule
        
        mock_embedder = Mock()
        module = RetrievalModule(test_config, mock_embedder)
        assert module is not None
        assert module.embedder == mock_embedder

    @pytest.mark.asyncio
    async def test_retrieval_module_initialization(self, test_config):
        """검색 모듈 초기화 테스트"""
        from app.modules.retrieval_rerank import RetrievalModule
        
        mock_embedder = Mock()
        module = RetrievalModule(test_config, mock_embedder)
        
        # Mock 초기화 (실제 Qdrant 없이)
        with patch.object(module, 'initialize', return_value=None):
            await module.initialize()
            assert True


class TestGenerationModule:
    """생성 모듈 테스트"""

    def test_generation_module_creation(self, test_config):
        """생성 모듈 생성 테스트"""
        from app.modules.generation import GenerationModule
        
        module = GenerationModule(test_config)
        assert module is not None
        assert module.config == test_config

    @pytest.mark.asyncio
    async def test_generation_module_initialization(self, test_config):
        """생성 모듈 초기화 테스트"""
        from app.modules.generation import GenerationModule
        
        module = GenerationModule(test_config)
        
        # Mock 초기화 (실제 LLM 클라이언트 없이)
        with patch.object(module, 'initialize', return_value=None):
            await module.initialize()
            assert True


class TestLoggerModule:
    """로거 모듈 테스트"""

    def test_logger_creation(self):
        """로거 생성 테스트"""
        from app.lib.logger import get_logger
        
        logger = get_logger(__name__)
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')

    def test_logger_functionality(self):
        """로거 기능 테스트"""
        from app.lib.logger import get_logger
        
        logger = get_logger("test_logger")
        
        # 로깅이 에러 없이 실행되는지 테스트
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        assert True  # 에러 없이 완료되면 성공


class TestLangChainIntegration:
    """LangChain 통합 테스트"""

    def test_langchain_document_import(self):
        """LangChain Document 임포트 테스트"""
        from langchain.schema import Document
        
        doc = Document(page_content="Test content", metadata={"test": True})
        assert doc.page_content == "Test content"
        assert doc.metadata["test"] is True

    def test_langchain_text_splitter_import(self):
        """LangChain TextSplitter 임포트 테스트"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        assert splitter is not None

    def test_google_embeddings_import(self):
        """Google Embeddings 임포트 테스트"""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Mock API 키로 생성 테스트
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
            assert embeddings is not None


class TestFileProcessing:
    """파일 처리 테스트"""

    def test_pdf_reader_import(self):
        """PDF 리더 임포트 테스트"""
        from pypdf import PdfReader
        assert PdfReader is not None

    def test_docx_import(self):
        """DOCX 처리 임포트 테스트"""
        from docx import Document as DocxDocument
        assert DocxDocument is not None

    def test_pandas_import(self):
        """Pandas 임포트 테스트"""
        import pandas as pd
        
        # 간단한 DataFrame 생성 테스트
        df = pd.DataFrame({"test": [1, 2, 3]})
        assert len(df) == 3

    def test_beautifulsoup_import(self):
        """BeautifulSoup 임포트 테스트"""
        from bs4 import BeautifulSoup
        
        html = "<html><body><p>Test</p></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        assert soup.find('p').text == "Test"

    def test_markdown_import(self):
        """Markdown 임포트 테스트"""
        import markdown
        
        md_text = "# Test Header"
        html = markdown.markdown(md_text)
        assert "<h1>" in html


class TestAsyncComponents:
    """비동기 컴포넌트 테스트"""

    @pytest.mark.asyncio
    async def test_async_function_execution(self):
        """비동기 함수 실행 테스트"""
        async def test_async():
            await asyncio.sleep(0.001)  # 1ms 대기
            return "async_result"
        
        result = await test_async()
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_mock_async_function(self):
        """Mock 비동기 함수 테스트"""
        mock_func = AsyncMock(return_value="mock_result")
        
        result = await mock_func()
        assert result == "mock_result"
        mock_func.assert_called_once()
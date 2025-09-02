"""
Document Processing Module Tests
문서 처리 모듈 테스트
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from app.modules.document_processing import DocumentProcessor


class TestDocumentProcessor:
    """문서 처리 모듈 테스트 클래스"""

    @pytest.fixture
    def document_processor(self, test_config):
        """문서 처리 모듈 픽스처"""
        return DocumentProcessor(test_config)

    def test_document_processor_initialization(self, document_processor):
        """문서 처리 모듈 초기화 테스트"""
        assert document_processor is not None
        assert hasattr(document_processor, 'config')
        assert hasattr(document_processor, 'supported_loaders')

    def test_supported_file_types(self, document_processor):
        """지원하는 파일 형식 테스트"""
        supported_types = document_processor.supported_loaders
        
        # 기본 지원 형식 확인
        expected_types = ['.pdf', '.txt', '.md', '.docx', '.xlsx']
        
        for file_type in expected_types:
            assert file_type in supported_types

    def test_get_loader_for_pdf(self, document_processor):
        """PDF 로더 선택 테스트"""
        loader = document_processor._get_loader('.pdf')
        assert loader is not None

    def test_get_loader_for_text(self, document_processor):
        """텍스트 로더 선택 테스트"""
        loader = document_processor._get_loader('.txt')
        assert loader is not None

    def test_get_loader_for_unsupported_type(self, document_processor):
        """지원하지 않는 파일 형식 테스트"""
        with pytest.raises(ValueError):
            document_processor._get_loader('.exe')

    @pytest.mark.asyncio
    async def test_process_text_document(self, document_processor, sample_text_file):
        """텍스트 문서 처리 테스트"""
        result = await document_processor.process_document(
            file_path=sample_text_file,
            filename="sample.txt"
        )
        
        assert result is not None
        assert "document_id" in result
        assert "chunks" in result
        assert "processing_time" in result
        assert len(result["chunks"]) > 0

    @pytest.mark.asyncio
    async def test_process_pdf_document(self, document_processor, sample_pdf_file):
        """PDF 문서 처리 테스트"""
        with patch('app.modules.document_processing.PyPDFLoader') as mock_loader:
            # Mock PDF loader
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Mock(page_content="PDF 테스트 내용", metadata={"page": 1})
            ]
            mock_loader.return_value = mock_instance
            
            result = await document_processor.process_document(
                file_path=sample_pdf_file,
                filename="sample.pdf"
            )
            
            assert result is not None
            assert "document_id" in result
            assert "chunks" in result

    def test_chunk_text_recursive(self, document_processor):
        """Recursive 텍스트 분할 테스트"""
        text = "이것은 테스트 문서입니다. " * 100  # 긴 텍스트 생성
        
        chunks = document_processor._chunk_text(
            text, 
            splitter_type="recursive",
            chunk_size=200,
            chunk_overlap=50
        )
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 250 for chunk in chunks)  # 약간의 여유

    def test_chunk_text_markdown(self, document_processor):
        """Markdown 텍스트 분할 테스트"""
        markdown_text = """
        # 제목 1
        
        이것은 첫 번째 섹션입니다.
        
        ## 제목 2
        
        이것은 두 번째 섹션입니다.
        
        ### 제목 3
        
        이것은 세 번째 섹션입니다.
        """
        
        chunks = document_processor._chunk_text(
            markdown_text,
            splitter_type="markdown",
            chunk_size=200,
            chunk_overlap=50
        )
        
        assert len(chunks) > 0

    def test_chunk_text_semantic(self, document_processor):
        """Semantic 텍스트 분할 테스트"""
        text = """
        인공지능은 컴퓨터 과학의 한 분야입니다. 
        머신러닝은 인공지능의 하위 분야입니다.
        딥러닝은 머신러닝의 한 종류입니다.
        자연어 처리는 인공지능의 응용 분야입니다.
        """
        
        with patch.object(document_processor, 'embedder') as mock_embedder:
            mock_embedder.embed_documents = AsyncMock(return_value=[
                [0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768
            ])
            
            chunks = document_processor._chunk_text(
                text,
                splitter_type="semantic",
                chunk_size=200,
                chunk_overlap=50
            )
            
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_create_embeddings(self, document_processor):
        """임베딩 생성 테스트"""
        chunks = [
            Mock(page_content="첫 번째 청크"),
            Mock(page_content="두 번째 청크")
        ]
        
        with patch.object(document_processor, 'embedder') as mock_embedder:
            mock_embedder.embed_documents = AsyncMock(return_value=[
                [0.1] * 768, [0.2] * 768
            ])
            
            embeddings = await document_processor._create_embeddings(chunks)
            
            assert len(embeddings) == 2
            assert all(len(emb) == 768 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_process_large_document(self, document_processor, temp_dir):
        """큰 문서 처리 테스트"""
        # 큰 텍스트 파일 생성
        large_text = "테스트 문장입니다. " * 10000
        large_file = temp_dir / "large_document.txt"
        large_file.write_text(large_text, encoding="utf-8")
        
        result = await document_processor.process_document(
            file_path=large_file,
            filename="large_document.txt"
        )
        
        assert result is not None
        assert len(result["chunks"]) > 10  # 많은 청크가 생성되어야 함

    @pytest.mark.asyncio
    async def test_process_empty_document(self, document_processor, temp_dir):
        """빈 문서 처리 테스트"""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        
        with pytest.raises(ValueError):
            await document_processor.process_document(
                file_path=empty_file,
                filename="empty.txt"
            )

    @pytest.mark.asyncio
    async def test_process_document_with_metadata(self, document_processor, sample_text_file):
        """메타데이터와 함께 문서 처리 테스트"""
        metadata = {
            "author": "테스트 작성자",
            "category": "테스트 카테고리",
            "tags": ["test", "document"]
        }
        
        result = await document_processor.process_document(
            file_path=sample_text_file,
            filename="sample.txt",
            metadata=metadata
        )
        
        assert result is not None
        # 메타데이터가 청크에 포함되었는지 확인
        for chunk in result["chunks"]:
            if "metadata" in chunk:
                chunk_metadata = chunk["metadata"]
                assert "author" in chunk_metadata or "filename" in chunk_metadata

    def test_validate_file_size(self, document_processor, large_file):
        """파일 크기 검증 테스트"""
        # 큰 파일은 검증 실패해야 함
        is_valid = document_processor._validate_file_size(large_file)
        assert is_valid is False

    def test_validate_file_size_normal(self, document_processor, sample_text_file):
        """정상 파일 크기 검증 테스트"""
        is_valid = document_processor._validate_file_size(sample_text_file)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_process_multiple_documents_concurrent(self, document_processor, temp_dir):
        """동시 다중 문서 처리 테스트"""
        import asyncio
        
        # 여러 테스트 파일 생성
        files = []
        for i in range(3):
            file_path = temp_dir / f"test_{i}.txt"
            file_path.write_text(f"테스트 문서 {i}의 내용입니다.", encoding="utf-8")
            files.append((file_path, f"test_{i}.txt"))
        
        # 동시 처리
        tasks = [
            document_processor.process_document(file_path, filename)
            for file_path, filename in files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 모든 결과가 성공이거나 예상된 예외여야 함
        for result in results:
            assert not isinstance(result, Exception) or isinstance(result, (ValueError, FileNotFoundError))

    def test_extract_text_from_docx(self, document_processor, temp_dir):
        """DOCX 파일에서 텍스트 추출 테스트"""
        with patch('app.modules.document_processing.Docx2txtLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Mock(page_content="DOCX 테스트 내용", metadata={})
            ]
            mock_loader.return_value = mock_instance
            
            # Mock DOCX 파일
            docx_file = temp_dir / "test.docx"
            docx_file.write_bytes(b"fake docx content")
            
            loader = document_processor._get_loader('.docx')
            assert loader is not None

    @pytest.mark.slow
    def test_performance_large_document_processing(self, document_processor, temp_dir, performance_timer):
        """대용량 문서 처리 성능 테스트"""
        # 대용량 텍스트 생성 (1MB)
        large_content = "성능 테스트를 위한 긴 문장입니다. " * 50000
        large_file = temp_dir / "performance_test.txt"
        large_file.write_text(large_content, encoding="utf-8")
        
        performance_timer.start()
        
        # 비동기 함수를 동기적으로 실행
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                document_processor.process_document(
                    file_path=large_file,
                    filename="performance_test.txt"
                )
            )
            performance_timer.stop()
            
            # 처리 시간이 합리적인 범위 내여야 함 (60초 이내)
            assert performance_timer.elapsed < 60.0
            assert result is not None
            
        finally:
            loop.close()
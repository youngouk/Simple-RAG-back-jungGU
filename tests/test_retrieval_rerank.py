"""
Retrieval and Reranking Module Tests
검색 및 리랭킹 모듈 테스트
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.modules.retrieval_rerank import RetrievalModule


class TestRetrievalModule:
    """검색 및 리랭킹 모듈 테스트 클래스"""

    @pytest.fixture
    def mock_embedder(self):
        """Mock 임베더 픽스처"""
        embedder = Mock()
        embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
        return embedder

    @pytest_asyncio.fixture
    async def retrieval_module(self, test_config, mock_embedder):
        """검색 모듈 픽스처"""
        module = RetrievalModule(test_config, mock_embedder)
        await module.initialize()
        return module

    @pytest.mark.asyncio
    async def test_retrieval_module_initialization(self, test_config, mock_embedder):
        """검색 모듈 초기화 테스트"""
        module = RetrievalModule(test_config, mock_embedder)
        await module.initialize()
        
        assert module is not None
        assert module.embedder == mock_embedder
        assert hasattr(module, 'qdrant_client')

    @pytest.mark.asyncio
    async def test_dense_search(self, retrieval_module):
        """Dense 검색 테스트"""
        query = "테스트 쿼리입니다"
        
        with patch.object(retrieval_module, 'qdrant_client') as mock_client:
            # Mock Qdrant 응답
            mock_response = Mock()
            mock_response.points = [
                Mock(
                    id="chunk-1",
                    payload={
                        "content": "테스트 문서 내용",
                        "metadata": {"document_id": "doc-1", "page": 1}
                    },
                    score=0.95
                )
            ]
            mock_client.search = AsyncMock(return_value=mock_response)
            
            results = await retrieval_module._dense_search(query, top_k=5)
            
            assert len(results) > 0
            assert results[0]["score"] == 0.95
            assert "content" in results[0]

    @pytest.mark.asyncio
    async def test_sparse_search(self, retrieval_module):
        """Sparse 검색 테스트"""
        query = "테스트 쿼리"
        
        with patch.object(retrieval_module, '_bm25_search') as mock_bm25:
            mock_bm25.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "테스트 문서 내용",
                    "score": 0.85,
                    "metadata": {"document_id": "doc-1"}
                }
            ]
            
            results = await retrieval_module._sparse_search(query, top_k=5)
            
            assert len(results) > 0
            assert results[0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_hybrid_search(self, retrieval_module):
        """하이브리드 검색 테스트"""
        query = "테스트 하이브리드 검색"
        
        with patch.object(retrieval_module, '_dense_search') as mock_dense, \
             patch.object(retrieval_module, '_sparse_search') as mock_sparse, \
             patch.object(retrieval_module, '_reciprocal_rank_fusion') as mock_rrf:
            
            # Mock dense 결과
            mock_dense.return_value = [
                {"chunk_id": "chunk-1", "content": "내용1", "score": 0.9},
                {"chunk_id": "chunk-2", "content": "내용2", "score": 0.8}
            ]
            
            # Mock sparse 결과
            mock_sparse.return_value = [
                {"chunk_id": "chunk-1", "content": "내용1", "score": 0.85},
                {"chunk_id": "chunk-3", "content": "내용3", "score": 0.75}
            ]
            
            # Mock RRF 결과
            mock_rrf.return_value = [
                {"chunk_id": "chunk-1", "content": "내용1", "score": 0.95},
                {"chunk_id": "chunk-2", "content": "내용2", "score": 0.82},
                {"chunk_id": "chunk-3", "content": "내용3", "score": 0.78}
            ]
            
            results = await retrieval_module.search(query, top_k=10)
            
            assert len(results) > 0
            assert results[0]["score"] >= results[1]["score"]  # 점수 순 정렬 확인

    def test_reciprocal_rank_fusion(self, retrieval_module):
        """Reciprocal Rank Fusion 테스트"""
        dense_results = [
            {"chunk_id": "chunk-1", "content": "내용1", "score": 0.9},
            {"chunk_id": "chunk-2", "content": "내용2", "score": 0.8},
            {"chunk_id": "chunk-3", "content": "내용3", "score": 0.7}
        ]
        
        sparse_results = [
            {"chunk_id": "chunk-2", "content": "내용2", "score": 0.85},
            {"chunk_id": "chunk-1", "content": "내용1", "score": 0.75},
            {"chunk_id": "chunk-4", "content": "내용4", "score": 0.65}
        ]
        
        fused_results = retrieval_module._reciprocal_rank_fusion(
            dense_results, sparse_results, k=60
        )
        
        assert len(fused_results) > 0
        
        # chunk-1과 chunk-2는 두 결과에 모두 있으므로 높은 점수를 가져야 함
        chunk_1_score = next(r["rrf_score"] for r in fused_results if r["chunk_id"] == "chunk-1")
        chunk_4_score = next(r["rrf_score"] for r in fused_results if r["chunk_id"] == "chunk-4")
        
        assert chunk_1_score > chunk_4_score

    @pytest.mark.asyncio
    async def test_reranking_with_jina(self, retrieval_module):
        """Jina 리랭킹 테스트"""
        query = "테스트 쿼리"
        initial_results = [
            {"chunk_id": "chunk-1", "content": "첫 번째 내용", "score": 0.8},
            {"chunk_id": "chunk-2", "content": "두 번째 내용", "score": 0.7},
            {"chunk_id": "chunk-3", "content": "세 번째 내용", "score": 0.6}
        ]
        
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock Jina API 응답
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 2, "relevance_score": 0.85},
                    {"index": 1, "relevance_score": 0.75}
                ]
            }
            mock_post.return_value = mock_response
            
            reranked = await retrieval_module._rerank_with_jina(query, initial_results)
            
            assert len(reranked) == len(initial_results)
            # 첫 번째 결과가 가장 높은 점수를 가져야 함
            assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]

    @pytest.mark.asyncio
    async def test_reranking_with_cohere(self, retrieval_module):
        """Cohere 리랭킹 테스트"""
        query = "테스트 쿼리"
        initial_results = [
            {"chunk_id": "chunk-1", "content": "첫 번째 내용", "score": 0.8},
            {"chunk_id": "chunk-2", "content": "두 번째 내용", "score": 0.7}
        ]
        
        with patch('cohere.AsyncClient') as mock_cohere:
            # Mock Cohere 클라이언트
            mock_client = Mock()
            mock_response = Mock()
            mock_response.results = [
                Mock(index=1, relevance_score=0.9),
                Mock(index=0, relevance_score=0.8)
            ]
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_cohere.return_value = mock_client
            
            reranked = await retrieval_module._rerank_with_cohere(query, initial_results)
            
            assert len(reranked) == len(initial_results)
            assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]

    @pytest.mark.asyncio
    async def test_semantic_similarity_reranking(self, retrieval_module):
        """의미 유사도 리랭킹 테스트"""
        query = "테스트 쿼리"
        initial_results = [
            {"chunk_id": "chunk-1", "content": "첫 번째 내용", "score": 0.8},
            {"chunk_id": "chunk-2", "content": "두 번째 내용", "score": 0.7}
        ]
        
        with patch.object(retrieval_module.embedder, 'embed_query') as mock_query_embed, \
             patch.object(retrieval_module.embedder, 'embed_documents') as mock_doc_embed:
            
            mock_query_embed.return_value = [0.1] * 768
            mock_doc_embed.return_value = [
                [0.2] * 768,  # 더 높은 유사도
                [0.05] * 768  # 더 낮은 유사도
            ]
            
            reranked = await retrieval_module._rerank_with_semantic_similarity(
                query, initial_results
            )
            
            assert len(reranked) == len(initial_results)
            # 의미적으로 더 유사한 문서가 먼저 와야 함
            assert reranked[0]["semantic_score"] >= reranked[1]["semantic_score"]

    @pytest.mark.asyncio
    async def test_search_with_filtering(self, retrieval_module):
        """필터링과 함께 검색 테스트"""
        query = "테스트 쿼리"
        filters = {
            "document_type": "pdf",
            "category": "technical"
        }
        
        with patch.object(retrieval_module, '_apply_metadata_filters') as mock_filter:
            mock_filter.return_value = [
                {"chunk_id": "chunk-1", "content": "필터링된 내용", "score": 0.9}
            ]
            
            results = await retrieval_module.search(
                query, top_k=5, filters=filters
            )
            
            assert len(results) > 0
            mock_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_query(self, retrieval_module):
        """빈 쿼리 검색 테스트"""
        with pytest.raises(ValueError):
            await retrieval_module.search("", top_k=5)

    @pytest.mark.asyncio
    async def test_search_no_results(self, retrieval_module):
        """검색 결과 없음 테스트"""
        query = "존재하지 않는 내용"
        
        with patch.object(retrieval_module, '_dense_search') as mock_dense, \
             patch.object(retrieval_module, '_sparse_search') as mock_sparse:
            
            mock_dense.return_value = []
            mock_sparse.return_value = []
            
            results = await retrieval_module.search(query, top_k=5)
            
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_performance(self, retrieval_module, performance_timer):
        """검색 성능 테스트"""
        query = "성능 테스트 쿼리"
        
        with patch.object(retrieval_module, '_dense_search') as mock_dense, \
             patch.object(retrieval_module, '_sparse_search') as mock_sparse:
            
            # 큰 결과 집합 Mock
            large_results = [
                {"chunk_id": f"chunk-{i}", "content": f"내용 {i}", "score": 0.9 - i*0.01}
                for i in range(100)
            ]
            mock_dense.return_value = large_results[:50]
            mock_sparse.return_value = large_results[25:75]
            
            performance_timer.start()
            results = await retrieval_module.search(query, top_k=20)
            performance_timer.stop()
            
            assert len(results) > 0
            # 검색은 5초 이내에 완료되어야 함
            assert performance_timer.elapsed < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, retrieval_module):
        """동시 검색 테스트"""
        import asyncio
        
        queries = [
            "첫 번째 쿼리",
            "두 번째 쿼리",
            "세 번째 쿼리"
        ]
        
        with patch.object(retrieval_module, '_dense_search') as mock_dense, \
             patch.object(retrieval_module, '_sparse_search') as mock_sparse:
            
            mock_dense.return_value = [
                {"chunk_id": "chunk-1", "content": "Mock 내용", "score": 0.9}
            ]
            mock_sparse.return_value = [
                {"chunk_id": "chunk-2", "content": "Mock 내용 2", "score": 0.8}
            ]
            
            # 동시 검색 실행
            tasks = [
                retrieval_module.search(query, top_k=5)
                for query in queries
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 모든 검색이 성공해야 함
            for result in results:
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_search_with_different_weights(self, retrieval_module):
        """다른 가중치로 하이브리드 검색 테스트"""
        query = "가중치 테스트"
        
        # 설정에서 가중치 변경
        retrieval_module.config["hybrid_search"] = {
            "dense_weight": 0.8,
            "sparse_weight": 0.2
        }
        
        with patch.object(retrieval_module, '_dense_search') as mock_dense, \
             patch.object(retrieval_module, '_sparse_search') as mock_sparse, \
             patch.object(retrieval_module, '_reciprocal_rank_fusion') as mock_rrf:
            
            mock_dense.return_value = [{"chunk_id": "chunk-1", "score": 0.9}]
            mock_sparse.return_value = [{"chunk_id": "chunk-1", "score": 0.8}]
            mock_rrf.return_value = [{"chunk_id": "chunk-1", "score": 0.85}]
            
            results = await retrieval_module.search(query, top_k=5)
            
            assert len(results) > 0
            # RRF가 호출되었는지 확인
            mock_rrf.assert_called_once()

    @pytest.mark.asyncio
    async def test_module_cleanup(self, retrieval_module):
        """모듈 정리 테스트"""
        # 모듈이 초기화되어 있는지 확인
        assert retrieval_module.qdrant_client is not None
        
        # 정리 실행
        await retrieval_module.close()
        
        # 정리 후 상태 확인 (구현에 따라 다를 수 있음)
        # 실제 구현에서는 연결이 닫혔는지 확인할 수 있음
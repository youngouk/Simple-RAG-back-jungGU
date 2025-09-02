"""
Retrieval and reranking module
검색 및 리랭킹 모듈 (Qdrant + 하이브리드 검색 + 리랭킹)
"""
import asyncio
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionInfo,
    PointStruct, Filter, FieldCondition, MatchValue,
    ScoredPoint, SparseVector, NamedVector, VectorParams,
    FusionQuery, Fusion, NearestQuery, Prefetch
)
from fastembed import SparseTextEmbedding

# Reranking clients
import httpx
from cohere import Client as CohereClient

from ..lib.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # 하위 호환성을 위해 속성으로도 접근 가능하게 설정
        for key, value in self.metadata.items():
            setattr(self, key, value)

class RetrievalModule:
    """검색 및 리랭킹 모듈"""
    
    def __init__(self, config: Dict[str, Any], embedder):
        self.config = config
        self.qdrant_config = config.get('qdrant', {})
        self.reranking_config = config.get('reranking', {})
        self.multi_query_config = config.get('multi_query', {})
        self.embeddings_config = config.get('embeddings', {})
        
        # Qdrant 클라이언트
        self.qdrant_client = None
        self.collection_name = self.qdrant_config.get('collection_name', 'documents')
        
        # 임베딩 모델들
        self.embedder = embedder  # Dense embedder
        self.sparse_embedder = None  # Sparse embedder (BM42)
        
        # 하이브리드 검색 설정
        self.dense_weight = self.qdrant_config.get('hybrid_search', {}).get('dense_weight', 0.6)
        self.sparse_weight = self.qdrant_config.get('hybrid_search', {}).get('sparse_weight', 0.4)
        self.hybrid_enabled = False
        
        # 리랭킹 클라이언트들
        self.rerankers = {}
        
        # 통계
        self.stats = {
            'total_searches': 0,
            'total_documents': 0,
            'vector_count': 0,
            'rerank_requests': 0,
            'hybrid_searches': 0
        }
    
    async def initialize(self):
        """모듈 초기화"""
        try:
            logger.info("Initializing retrieval module...")
            
            # 1. Sparse embedder 초기화
            await self._init_sparse_embedder()
            
            # 2. Qdrant 클라이언트 초기화
            await self._init_qdrant()
            
            # 3. 컬렉션 초기화
            await self._init_collection()
            
            # 4. 리랭커 초기화
            await self._init_rerankers()
            
            logger.info(f"Retrieval module initialized successfully (hybrid: {self.hybrid_enabled})")
            
        except Exception as e:
            logger.error(f"Retrieval module initialization failed: {e}")
            raise
    
    async def close(self):
        """모듈 정리"""
        try:
            if self.qdrant_client:
                self.qdrant_client.close()
            logger.info("Retrieval module closed")
        except Exception as e:
            logger.error(f"Retrieval module close error: {e}")
    
    async def _init_qdrant(self):
        """Qdrant 클라이언트 초기화"""
        # URL 우선 사용 (클라우드 서비스의 경우)
        qdrant_url = self.qdrant_config.get('url')
        api_key = self.qdrant_config.get('api_key')
        
        if qdrant_url and qdrant_url.startswith('https://'):
            # 클라우드 Qdrant 서비스
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=api_key,
                timeout=30
            )
            logger.info(f"Qdrant client connected to cloud service: {qdrant_url}")
        else:
            # 로컬 Qdrant 서버
            host = self.qdrant_config.get('host', 'localhost')
            port = self.qdrant_config.get('port', 6333)
            
            # HTTP vs gRPC 선택
            prefer_grpc = self.qdrant_config.get('prefer_grpc', False)
            
            if prefer_grpc:
                self.qdrant_client = QdrantClient(host=host, port=port, prefer_grpc=True)
            else:
                self.qdrant_client = QdrantClient(url=f"http://{host}:{port}")
            
            logger.info(f"Qdrant client connected to local server: {host}:{port}")
        
        # 연결 테스트
        await asyncio.to_thread(self.qdrant_client.get_collections)
        logger.info("Qdrant connection test successful")
    
    async def _init_sparse_embedder(self):
        """Sparse embedder 초기화"""
        try:
            sparse_model = self.embeddings_config.get('sparse_model', 'Qdrant/bm42-all-minilm-l6-v2-attentions')
            self.sparse_embedder = SparseTextEmbedding(model_name=sparse_model)
            self.hybrid_enabled = True
            logger.info(f"Sparse embedder initialized: {sparse_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize sparse embedder: {e}")
            self.sparse_embedder = None
            self.hybrid_enabled = False
    
    async def _init_collection(self):
        """컬렉션 초기화 (하이브리드 벡터 지원)"""
        try:
            # 컬렉션 존재 확인
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating hybrid collection: {self.collection_name}")
                
                # Dense 임베딩 차원 계산
                test_embedding = await asyncio.to_thread(
                    self.embedder.embed_query, "test"
                )
                dense_vector_size = len(test_embedding)
                
                # 벡터 설정 구성
                vectors_config = {
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=Distance.COSINE
                    )
                }
                
                # Sparse 벡터 추가 (사용 가능한 경우)
                if self.hybrid_enabled:
                    from qdrant_client.models import SparseVectorParams
                    vectors_config["sparse"] = SparseVectorParams()
                    logger.info("Adding sparse vector support to collection")
                
                # 컬렉션 생성
                await asyncio.to_thread(
                    self.qdrant_client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                
                logger.info(f"Collection {self.collection_name} created with dense size {dense_vector_size} and hybrid support")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
            # 컬렉션 정보 업데이트
            await self._update_collection_stats()
            
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise
    
    async def _init_rerankers(self):
        """리랭커 초기화"""
        if not self.reranking_config.get('enabled', False):
            logger.info("Reranking disabled")
            return
        
        providers = self.reranking_config.get('providers', {})
        
        # Cohere 리랭커
        if 'cohere' in providers:
            cohere_config = providers['cohere']
            if cohere_config.get('api_key'):
                self.rerankers['cohere'] = CohereClient(
                    api_key=cohere_config['api_key']
                )
                logger.info("Cohere reranker initialized")
        
        # Jina 리랭커 (HTTP API)
        if 'jina' in providers:
            jina_config = providers['jina']
            if jina_config.get('api_key'):
                self.rerankers['jina'] = {
                    'api_key': jina_config['api_key'],
                    'model': jina_config.get('model', 'jina-reranker-v1-base-en'),
                    'endpoint': jina_config.get('endpoint', 'https://api.jina.ai/v1/rerank')
                }
                logger.info("Jina reranker initialized")
        
        logger.info(f"Initialized {len(self.rerankers)} rerankers")
    
    async def _update_collection_stats(self):
        """컬렉션 통계 업데이트"""
        try:
            collection_info = await asyncio.to_thread(
                self.qdrant_client.get_collection,
                collection_name=self.collection_name
            )
            
            self.stats['vector_count'] = collection_info.vectors_count or 0
            self.stats['total_documents'] = collection_info.points_count or 0
            
        except Exception as e:
            logger.warning(f"Failed to update collection stats: {e}")
    
    async def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """문서 추가 (하이브리드 벡터 지원)"""
        if not embedded_chunks:
            return True
        
        try:
            logger.info(f"Adding {len(embedded_chunks)} documents to collection")
            
            # Point 객체 생성
            points = []
            for i, chunk in enumerate(embedded_chunks):
                # UUID 기반 point ID 생성
                import uuid
                point_id = str(uuid.uuid4())
                
                # 벡터 구성
                vectors = {"dense": chunk['dense_embedding']}
                
                # Sparse 벡터 추가 (있는 경우)
                if 'sparse_embedding' in chunk and self.hybrid_enabled:
                    sparse_data = chunk['sparse_embedding']
                    vectors["sparse"] = SparseVector(
                        indices=sparse_data['indices'],
                        values=sparse_data['values']
                    )
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={
                        'content': chunk['content'],
                        'metadata': chunk['metadata']
                    }
                ))
            
            # 배치로 업로드
            batch_size = self.qdrant_config.get('batch_size', 100)
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                await asyncio.to_thread(
                    self.qdrant_client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
                
                logger.debug(f"Uploaded batch {i//batch_size + 1}/{math.ceil(len(points)/batch_size)}")
            
            # 통계 업데이트
            await self._update_collection_stats()
            
            logger.info(f"Successfully added {len(embedded_chunks)} documents with hybrid vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def search(self, query: str, options: Dict[str, Any] = None) -> List[SearchResult]:
        """검색 실행 (Qdrant 네이티브 하이브리드 검색)"""
        options = options or {}
        limit = options.get('limit', 20)
        min_score = options.get('min_score', 0.3)
        
        self.stats['total_searches'] += 1
        
        try:
            logger.debug(f"Searching for: {query[:50]}...")
            
            # 임시로 Dense 검색만 사용 (하이브리드 검색에서 RRF 점수 문제 때문에)
            if False:  # self.hybrid_enabled:
                # 하이브리드 검색 실행
                results = await self._hybrid_search(query, limit, min_score)
                self.stats['hybrid_searches'] += 1
                logger.info(f"Hybrid search completed: {len(results)} results")
            else:
                # Dense 검색만 실행
                query_embedding = await asyncio.to_thread(
                    self.embedder.embed_query, query
                )
                results = await self._dense_search(query_embedding, limit=limit)
                
                # 점수 필터링
                results = [
                    result for result in results 
                    if result.score >= min_score
                ]
                logger.info(f"Dense search completed: {len(results)} results")
            
            # 점수 로깅 (디버깅용)
            for i, result in enumerate(results[:3]):  # 최상위 3개만 로깅
                logger.info(f"Result {i+1}: score={result.score:.4f}, content preview: {result.content[:50]}...")
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _hybrid_search(self, query: str, limit: int, min_score: float) -> List[SearchResult]:
        """Qdrant 네이티브 하이브리드 검색"""
        try:
            # 쿼리 임베딩 생성
            dense_embedding = await asyncio.to_thread(
                self.embedder.embed_query, query
            )
            
            # Sparse 임베딩 생성
            sparse_results = await asyncio.to_thread(
                list, self.sparse_embedder.embed([query])
            )
            sparse_embedding = sparse_results[0] if sparse_results else None
            
            if sparse_embedding is None:
                logger.warning("Failed to generate sparse embedding, falling back to dense search")
                return await self._dense_search(dense_embedding, limit)
            
            # Sparse vector 생성
            sparse_vector = SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            # Qdrant 네이티브 하이브리드 검색 실행
            logger.debug("Executing Qdrant native hybrid search with RRF fusion")
            query_response = await asyncio.to_thread(
                self.qdrant_client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=NearestQuery(nearest=dense_embedding),
                        using="dense",
                        limit=limit * 2
                    ),
                    Prefetch(
                        query=NearestQuery(nearest=sparse_vector),
                        using="sparse", 
                        limit=limit * 2
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                score_threshold=min_score
            )
            
            # 결과 처리
            results = []
            for point in query_response.points:
                results.append(SearchResult(
                    id=str(point.id),
                    content=point.payload['content'],
                    score=point.score,
                    metadata=point.payload['metadata']
                ))
            
            logger.info(f"Native hybrid search completed: {len(results)} results (RRF fusion)")
            return results
            
        except Exception as e:
            logger.error(f"Native hybrid search failed: {e}")
            logger.info("Falling back to legacy hybrid search implementation")
            
            # 실패 시 기존 방식으로 폴백
            try:
                return await self._legacy_hybrid_search(query, limit, min_score)
            except Exception as fallback_error:
                logger.error(f"Legacy hybrid search also failed: {fallback_error}")
                # 최종적으로 dense 검색으로 폴백
                return await self._dense_search(
                    await asyncio.to_thread(self.embedder.embed_query, query), limit
                )
    
    async def _legacy_hybrid_search(self, query: str, limit: int, min_score: float) -> List[SearchResult]:
        """기존 방식의 하이브리드 검색 (폴백용)"""
        try:
            logger.info("Using legacy hybrid search implementation")
            
            # 쿼리 임베딩 생성
            dense_embedding = await asyncio.to_thread(
                self.embedder.embed_query, query
            )
            
            # Sparse 임베딩 생성
            sparse_results = await asyncio.to_thread(
                list, self.sparse_embedder.embed([query])
            )
            sparse_embedding = sparse_results[0] if sparse_results else None
            
            if sparse_embedding is None:
                logger.warning("Failed to generate sparse embedding in legacy mode")
                return await self._dense_search(dense_embedding, limit)
            
            # 별도 검색 실행 (기존 방식)
            dense_results = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name=self.collection_name,
                query_vector=("dense", dense_embedding),
                limit=limit * 2,
                with_payload=True
            )
            
            sparse_results = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name=self.collection_name,
                query_vector=("sparse", SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist()
                )),
                limit=limit * 2,
                with_payload=True
            )
            
            # RRF 융합
            fused_results = self._rrf_fusion(dense_results, sparse_results, limit)
            
            # 점수 필터링
            filtered_results = [
                result for result in fused_results 
                if result.score >= min_score
            ]
            
            logger.info(f"Legacy hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(fused_results)} fused -> {len(filtered_results)} filtered")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Legacy hybrid search failed: {e}")
            # 최종 폴백: dense 검색만
            return await self._dense_search(
                await asyncio.to_thread(self.embedder.embed_query, query), limit
            )
    
    async def _dense_search(self, query_embedding: List[float], limit: int) -> List[SearchResult]:
        """Dense 벡터 검색"""
        try:
            search_result = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                limit=limit,
                with_payload=True
            )
            
            results = []
            for point in search_result:
                results.append(SearchResult(
                    id=str(point.id),
                    content=point.payload['content'],
                    score=point.score,
                    metadata=point.payload['metadata']
                ))
            
            logger.info(f"Dense search processed {len(search_result)} raw results into {len(results)} SearchResult objects")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _rrf_fusion(self, dense_points: List, sparse_points: List, limit: int) -> List[SearchResult]:
        """Reciprocal Rank Fusion으로 dense와 sparse 결과 융합"""
        try:
            k = 60  # RRF 상수
            doc_scores = {}
            
            # Dense 결과 처리
            for rank, point in enumerate(dense_points):
                point_id = str(point.id)
                if point_id not in doc_scores:
                    doc_scores[point_id] = {
                        'point': point,
                        'rrf_score': 0
                    }
                
                rrf_score = 1 / (k + rank + 1)
                doc_scores[point_id]['rrf_score'] += rrf_score * self.dense_weight
            
            # Sparse 결과 처리
            for rank, point in enumerate(sparse_points):
                point_id = str(point.id)
                if point_id not in doc_scores:
                    doc_scores[point_id] = {
                        'point': point,
                        'rrf_score': 0
                    }
                
                rrf_score = 1 / (k + rank + 1)
                doc_scores[point_id]['rrf_score'] += rrf_score * self.sparse_weight
            
            # 점수순 정렬
            sorted_docs = sorted(
                doc_scores.values(),
                key=lambda x: x['rrf_score'],
                reverse=True
            )
            
            # SearchResult 객체 생성
            results = []
            for doc_data in sorted_docs[:limit]:
                point = doc_data['point']
                results.append(SearchResult(
                    id=str(point.id),
                    content=point.payload['content'],
                    score=doc_data['rrf_score'],
                    metadata=point.payload['metadata']
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"RRF fusion failed: {e}")
            # 실패 시 dense 결과만 반환
            results = []
            for point in dense_points[:limit]:
                results.append(SearchResult(
                    id=str(point.id),
                    content=point.payload['content'],
                    score=point.score,
                    metadata=point.payload['metadata']
                ))
            return results
    
    
    async def rerank(self, query: str, search_results: List[SearchResult], 
                    options: Dict[str, Any] = None) -> List[SearchResult]:
        """리랭킹 실행"""
        if not self.reranking_config.get('enabled', False) or not search_results:
            return search_results
        
        options = options or {}
        top_k = options.get('top_k', 5)
        min_score = options.get('min_score', 0.4)  # 최소 유사도 임계값 추가
        provider = self.reranking_config.get('default_provider', 'cohere')
        
        self.stats['rerank_requests'] += 1
        
        try:
            logger.debug(f"Reranking {len(search_results)} results with {provider}")
            
            reranked_results = None
            if provider == 'cohere' and 'cohere' in self.rerankers:
                reranked_results = await self._rerank_cohere(query, search_results, top_k)
            elif provider == 'jina' and 'jina' in self.rerankers:
                reranked_results = await self._rerank_jina(query, search_results, top_k)
            else:
                logger.warning(f"Reranker {provider} not available, skipping reranking")
                return search_results
            
            # 리랭킹 후 최소 점수 필터링 적용
            if reranked_results:
                filtered_results = [
                    result for result in reranked_results 
                    if result.score >= min_score
                ]
                logger.info(f"Post-reranking filtering: {len(reranked_results)} -> {len(filtered_results)} results (min_score={min_score})")
                return filtered_results
                
            return search_results
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return search_results  # 실패 시 원본 결과 반환
    
    async def _rerank_cohere(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Cohere 리랭킹"""
        try:
            cohere_client = self.rerankers['cohere']
            
            # 문서 텍스트 추출
            documents = [result.content for result in results]
            
            # 리랭킹 실행
            rerank_response = await asyncio.to_thread(
                cohere_client.rerank,
                model="rerank-multilingual-v2.0",
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents))
            )
            
            # 결과 재구성
            reranked_results = []
            for rank_result in rerank_response.results:
                original_result = results[rank_result.index]
                original_result.score = rank_result.relevance_score
                reranked_results.append(original_result)
            
            logger.debug(f"Cohere reranking completed: {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return results
    
    async def _rerank_jina(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Jina 리랭킹"""
        try:
            jina_config = self.rerankers['jina']
            
            # HTTP 요청 데이터
            documents = [result.content for result in results]
            
            request_data = {
                "model": jina_config['model'],
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents))
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jina_config['api_key']}"
            }
            
            # HTTP 요청 실행
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    jina_config['endpoint'],
                    json=request_data,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                rerank_response = response.json()
            
            # 결과 재구성
            reranked_results = []
            for rank_result in rerank_response['results']:
                original_result = results[rank_result['index']]
                original_result.score = rank_result['relevance_score']
                reranked_results.append(original_result)
            
            logger.debug(f"Jina reranking completed: {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Jina reranking failed: {e}")
            return results
    
    async def list_documents(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """문서 목록 조회"""
        try:
            offset = (page - 1) * page_size
            
            # 스크롤로 문서 조회
            scroll_result = await asyncio.to_thread(
                self.qdrant_client.scroll,
                collection_name=self.collection_name,
                limit=page_size,
                offset=offset,
                with_payload=True
            )
            
            points, next_page_offset = scroll_result
            
            documents = []
            for point in points:
                metadata = point.payload['metadata']
                documents.append({
                    'id': str(point.id),
                    'filename': metadata.get('source_file', 'unknown'),
                    'file_type': metadata.get('file_type', 'unknown'),
                    'file_size': metadata.get('file_size', 0),
                    'upload_date': metadata.get('load_timestamp', 0),
                    'chunk_count': metadata.get('total_chunks', 1)
                })
            
            return {
                'documents': documents,
                'total_count': self.stats['total_documents'],
                'page': page,
                'page_size': page_size,
                'has_next': next_page_offset is not None
            }
            
        except Exception as e:
            logger.error(f"List documents failed: {e}")
            return {
                'documents': [],
                'total_count': 0,
                'page': page,
                'page_size': page_size,
                'has_next': False
            }
    
    async def delete_document(self, document_id: str):
        """문서 삭제"""
        try:
            await asyncio.to_thread(
                self.qdrant_client.delete,
                collection_name=self.collection_name,
                points_selector=[document_id]
            )
            
            await self._update_collection_stats()
            logger.info(f"Document deleted: {document_id}")
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        await self._update_collection_stats()
        
        return {
            **self.stats,
            'collection_name': self.collection_name,
            'hybrid_search_enabled': self.hybrid_enabled,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'reranking_enabled': self.reranking_config.get('enabled', False),
            'available_rerankers': list(self.rerankers.keys()),
            'sparse_embedder_available': self.sparse_embedder is not None
        }
    
    async def clear_cache(self):
        """캐시 클리어 (현재 구현에서는 통계만 업데이트)"""
        await self._update_collection_stats()
        logger.info("Retrieval cache cleared (stats updated)")
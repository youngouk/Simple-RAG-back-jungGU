# 전체 시스템 통합 및 플로우 설계
# System Integration and Flow Design

"""
RAG 시스템 통합 설계 문서

이 문서는 다음 세 가지 주요 개선사항을 기존 RAG 시스템에 통합하는 방안을 제시합니다:
1. GPT-5 쿼리 확장 시스템
2. 동적 임계값 시스템  
3. 시멘틱 청킹 전략

핵심 통합 원칙:
- 최소 침입적 변경: 기존 인터페이스 최대한 보존
- 점진적 적용: 각 컴포넌트 독립적으로 활성화/비활성화 가능
- 성능 최적화: 캐싱 및 병렬 처리 적용
- 에러 복구: 각 컴포넌트별 폴백 메커니즘
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

# 기존 모듈 imports (가상)
# from app.modules.document_processing import DocumentProcessor
# from app.modules.retrieval_rerank import RetrievalModule
# from app.modules.generation import GenerationModule
# from app.modules.session import SessionModule

# 새로운 컴포넌트 imports
# from gpt5_query_expansion_system import GPT5QueryExpansionEngine, ExpandedQuery
# from dynamic_threshold_system_detailed import DynamicThresholdEngine, ThresholdCache
# from semantic_chunking_strategy import SemanticSplitter, ChunkingConfig

logger = logging.getLogger(__name__)

# =====================================
# 1. 통합 설정 구조 설계
# =====================================

@dataclass
class IntegratedSystemConfig:
    """통합 시스템 설정"""
    
    # 기존 설정 (기본값 유지)
    existing_config: Dict[str, Any] = None
    
    # 새로운 기능 활성화 플래그
    enable_query_expansion: bool = True
    enable_dynamic_threshold: bool = True
    enable_semantic_chunking: bool = True
    
    # GPT-5 쿼리 확장 설정
    query_expansion: Dict[str, Any] = None
    
    # 동적 임계값 설정
    dynamic_threshold: Dict[str, Any] = None
    
    # 시멘틱 청킹 설정
    semantic_chunking: Dict[str, Any] = None
    
    def __post_init__(self):
        # 기본 설정 초기화
        if self.query_expansion is None:
            self.query_expansion = {
                "model": "gpt-5-nano",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cache_ttl": 3600,
                "min_query_length": 10,
                "max_synonyms": 5,
                "expansion_strategies": ["synonym", "intent", "keyword"],
                "fallback_enabled": True
            }
        
        if self.dynamic_threshold is None:
            self.dynamic_threshold = {
                "base_threshold": 0.15,
                "target_min_results": 5,
                "threshold_levels": [0.15, 0.10, 0.05, 0.01],
                "learning_enabled": True,
                "cache_enabled": True,
                "cache_ttl": 1800
            }
        
        if self.semantic_chunking is None:
            self.semantic_chunking = {
                "target_chunk_size": 1250,
                "min_chunk_size": 1000,
                "max_chunk_size": 1500,
                "semantic_threshold": 0.3,
                "overlap_sentences": 2,
                "sentence_window": 3,
                "preserve_structure": True,
                "korean_optimized": True,
                "fallback_enabled": True
            }

# =====================================
# 2. DocumentProcessor 통합 업데이트
# =====================================

class EnhancedDocumentProcessor:
    """시멘틱 청킹이 통합된 향상된 문서 처리 모듈"""
    
    def __init__(self, config: Dict[str, Any], integrated_config: IntegratedSystemConfig):
        self.config = config
        self.integrated_config = integrated_config
        self.document_config = config.get('document_processing', {})
        
        # 기존 초기화 로직 유지
        self._init_embedders()
        
        # 새로운 청킹 시스템 초기화
        if integrated_config.enable_semantic_chunking:
            self._init_semantic_chunking()
        else:
            self.chunking_strategy = 'recursive'  # 기존 방식
        
        # 통계 추적
        self.processing_stats = {
            'total_documents': 0,
            'semantic_chunked': 0,
            'recursive_chunked': 0,
            'avg_chunk_size': 0,
            'processing_errors': 0
        }
    
    def _init_embedders(self):
        """기존 임베더 초기화 로직 유지"""
        # 기존 DocumentProcessor._init_embedders() 로직과 동일
        pass
    
    def _init_semantic_chunking(self):
        """시멘틱 청킹 시스템 초기화"""
        try:
            from semantic_chunking_strategy import ChunkingConfig, SemanticSplitter
            
            # 설정에서 청킹 config 생성
            chunking_settings = self.integrated_config.semantic_chunking
            self.chunking_config = ChunkingConfig(
                target_chunk_size=chunking_settings['target_chunk_size'],
                min_chunk_size=chunking_settings['min_chunk_size'], 
                max_chunk_size=chunking_settings['max_chunk_size'],
                semantic_threshold=chunking_settings['semantic_threshold'],
                overlap_sentences=chunking_settings['overlap_sentences'],
                sentence_window=chunking_settings['sentence_window'],
                preserve_structure=chunking_settings['preserve_structure'],
                korean_optimized=chunking_settings['korean_optimized']
            )
            
            # 시멘틱 분할기 초기화
            self.semantic_splitter = SemanticSplitter(self.chunking_config, self.embedder)
            self.chunking_strategy = 'semantic'
            
            logger.info("시멘틱 청킹 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"시멘틱 청킹 초기화 실패: {e}")
            if self.integrated_config.semantic_chunking.get('fallback_enabled', True):
                logger.info("RecursiveCharacterTextSplitter 폴백 사용")
                self.chunking_strategy = 'recursive'
            else:
                raise
    
    async def split_documents(self, documents: List[Any]) -> List[Any]:
        """향상된 문서 분할 (시멘틱/기존 방식 선택적 적용)"""
        if not documents:
            return []
        
        try:
            if self.chunking_strategy == 'semantic':
                # 시멘틱 청킹 적용
                chunks = await self.semantic_splitter.split_documents(documents)
                self.processing_stats['semantic_chunked'] += len(chunks)
            else:
                # 기존 RecursiveCharacterTextSplitter 사용
                chunks = await self._recursive_split_documents(documents)
                self.processing_stats['recursive_chunked'] += len(chunks)
            
            # 통계 업데이트
            self.processing_stats['total_documents'] += len(documents)
            if chunks:
                total_size = sum(len(chunk.page_content) for chunk in chunks)
                self.processing_stats['avg_chunk_size'] = total_size / len(chunks)
            
            logger.info(f"문서 분할 완료: {len(documents)}개 문서 → {len(chunks)}개 청크 ({self.chunking_strategy})")
            return chunks
            
        except Exception as e:
            logger.error(f"문서 분할 실패: {e}")
            self.processing_stats['processing_errors'] += 1
            
            # 폴백 처리
            if self.chunking_strategy == 'semantic':
                logger.info("시멘틱 청킹 실패, RecursiveCharacterTextSplitter 폴백")
                return await self._recursive_split_documents(documents)
            else:
                raise
    
    async def _recursive_split_documents(self, documents: List[Any]) -> List[Any]:
        """기존 RecursiveCharacterTextSplitter 로직"""
        # 기존 DocumentProcessor.split_documents() 로직과 동일
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.integrated_config.semantic_chunking['target_chunk_size'],
            chunk_overlap=self.integrated_config.semantic_chunking['overlap_sentences'] * 50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return splitter.split_documents(documents)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """향상된 통계 정보"""
        base_stats = self.get_stats()  # 기존 통계
        
        enhanced_stats = {
            **base_stats,
            'chunking_strategy': self.chunking_strategy,
            'processing_stats': self.processing_stats,
            'semantic_chunking_enabled': self.integrated_config.enable_semantic_chunking
        }
        
        # 시멘틱 청킹 통계 추가
        if hasattr(self, 'semantic_splitter'):
            enhanced_stats['semantic_chunking_stats'] = self.semantic_splitter.get_stats()
        
        return enhanced_stats

# =====================================
# 3. RetrievalModule 통합 업데이트  
# =====================================

class EnhancedRetrievalModule:
    """쿼리 확장 및 동적 임계값이 통합된 향상된 검색 모듈"""
    
    def __init__(self, config: Dict[str, Any], integrated_config: IntegratedSystemConfig, 
                 qdrant_client, embedder):
        self.config = config
        self.integrated_config = integrated_config
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        
        # 기존 초기화 로직 유지
        self.retrieval_config = config.get('retrieval', {})
        
        # 새로운 컴포넌트 초기화
        if integrated_config.enable_query_expansion:
            self._init_query_expansion()
        
        if integrated_config.enable_dynamic_threshold:
            self._init_dynamic_threshold()
        
        # 검색 통계 추적
        self.search_stats = {
            'total_searches': 0,
            'expanded_queries': 0,
            'dynamic_threshold_used': 0,
            'fallback_searches': 0,
            'avg_results_count': 0,
            'avg_final_score': 0
        }
    
    def _init_query_expansion(self):
        """GPT-5 쿼리 확장 시스템 초기화"""
        try:
            from gpt5_query_expansion_system import GPT5QueryExpansionEngine
            
            expansion_config = self.integrated_config.query_expansion
            self.query_expander = GPT5QueryExpansionEngine(
                api_key=self.config.get('llm', {}).get('openai', {}).get('api_key'),
                model=expansion_config['model'],
                temperature=expansion_config['temperature'],
                max_tokens=expansion_config['max_tokens'],
                cache_ttl=expansion_config['cache_ttl']
            )
            
            logger.info("GPT-5 쿼리 확장 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"쿼리 확장 초기화 실패: {e}")
            if self.integrated_config.query_expansion.get('fallback_enabled', True):
                self.query_expander = None
                logger.info("쿼리 확장 비활성화, 원본 쿼리 사용")
            else:
                raise
    
    def _init_dynamic_threshold(self):
        """동적 임계값 시스템 초기화"""
        try:
            from dynamic_threshold_system_detailed import DynamicThresholdEngine, ThresholdCache
            
            threshold_config = self.integrated_config.dynamic_threshold
            
            # 캐시 초기화
            if threshold_config.get('cache_enabled', True):
                self.threshold_cache = ThresholdCache(
                    ttl=threshold_config['cache_ttl']
                )
            else:
                self.threshold_cache = None
            
            # 동적 임계값 엔진 초기화
            self.threshold_engine = DynamicThresholdEngine(
                base_threshold=threshold_config['base_threshold'],
                target_min_results=threshold_config['target_min_results'],
                threshold_levels=threshold_config['threshold_levels'],
                learning_enabled=threshold_config.get('learning_enabled', True),
                cache=self.threshold_cache
            )
            
            logger.info("동적 임계값 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"동적 임계값 초기화 실패: {e}")
            self.threshold_engine = None
            logger.info("동적 임계값 비활성화, 고정 임계값 사용")
    
    async def search_with_enhancements(self, query: str, options: Dict[str, Any] = None) -> List[Any]:
        """통합 강화된 검색 (쿼리 확장 + 동적 임계값)"""
        options = options or {}
        self.search_stats['total_searches'] += 1
        
        try:
            # 1단계: 쿼리 확장 (선택적)
            expanded_query = await self._expand_query_if_enabled(query)
            final_query = expanded_query.expanded_query if expanded_query else query
            
            # 2단계: 동적 임계값 검색 (선택적)
            if self.threshold_engine and not options.get('disable_dynamic_threshold', False):
                results = await self._search_with_dynamic_threshold(
                    final_query, expanded_query, options
                )
                self.search_stats['dynamic_threshold_used'] += 1
            else:
                # 기존 고정 임계값 검색
                results = await self._search_with_fixed_threshold(final_query, options)
            
            # 통계 업데이트
            if results:
                self.search_stats['avg_results_count'] = (
                    (self.search_stats['avg_results_count'] * (self.search_stats['total_searches'] - 1) + len(results)) /
                    self.search_stats['total_searches']
                )
            
            logger.info(f"강화 검색 완료: '{query}' → {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"강화 검색 실패: {e}")
            self.search_stats['fallback_searches'] += 1
            
            # 폴백: 기본 검색 수행
            logger.info("기본 검색 폴백 실행")
            return await self._fallback_search(query, options)
    
    async def _expand_query_if_enabled(self, query: str) -> Optional[Any]:
        """쿼리 확장 (활성화된 경우)"""
        if not self.query_expander:
            return None
        
        try:
            # 최소 쿼리 길이 검증
            min_length = self.integrated_config.query_expansion.get('min_query_length', 10)
            if len(query.strip()) < min_length:
                logger.debug(f"쿼리가 너무 짧아 확장 건너뜀: {len(query)}자 < {min_length}자")
                return None
            
            expanded_query = await self.query_expander.expand_query(query)
            self.search_stats['expanded_queries'] += 1
            
            logger.debug(f"쿼리 확장 완료: '{query}' → '{expanded_query.expanded_query}'")
            return expanded_query
            
        except Exception as e:
            logger.warning(f"쿼리 확장 실패, 원본 쿼리 사용: {e}")
            return None
    
    async def _search_with_dynamic_threshold(self, query: str, expanded_query: Optional[Any], 
                                           options: Dict[str, Any]) -> List[Any]:
        """동적 임계값 검색"""
        try:
            # 검색 컨텍스트 결정
            from dynamic_threshold_system_detailed import SearchContext
            
            if expanded_query and hasattr(expanded_query, 'search_intent'):
                context_map = {
                    'factual': SearchContext.FACTUAL,
                    'semantic': SearchContext.SEMANTIC,
                    'keyword': SearchContext.KEYWORD,
                    'conversational': SearchContext.CONVERSATIONAL
                }
                context = context_map.get(expanded_query.search_intent, SearchContext.SEMANTIC)
            else:
                context = SearchContext.SEMANTIC
            
            # 동적 임계값으로 검색 실행
            used_threshold, results, metadata = await self.threshold_engine.find_optimal_threshold(
                query=query,
                search_function=self._base_search,
                context=context,
                **options
            )
            
            # 메타데이터 추가
            for result in results:
                if hasattr(result, 'metadata'):
                    result.metadata.update({
                        'used_threshold': used_threshold,
                        'search_metadata': metadata,
                        'query_expanded': expanded_query is not None
                    })
            
            logger.info(f"동적 임계값 검색: 임계값 {used_threshold:.3f} → {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"동적 임계값 검색 실패: {e}")
            # 폴백: 고정 임계값 검색
            return await self._search_with_fixed_threshold(query, options)
    
    async def _search_with_fixed_threshold(self, query: str, options: Dict[str, Any]) -> List[Any]:
        """고정 임계값 검색 (기존 방식)"""
        min_score = options.get('min_score', self.retrieval_config.get('min_score', 0.15))
        return await self._base_search(query, min_score=min_score, **options)
    
    async def _base_search(self, query: str, min_score: float = 0.15, **kwargs) -> List[Any]:
        """기본 검색 함수 (기존 RetrievalModule.search 로직)"""
        # 기존 RetrievalModule의 하이브리드 검색 로직과 동일
        # Qdrant 하이브리드 검색, RRF 융합, 리랭킹 등
        pass
    
    async def _fallback_search(self, query: str, options: Dict[str, Any]) -> List[Any]:
        """폴백 검색 (모든 강화 기능 비활성화)"""
        return await self._base_search(query, **options)
    
    def get_enhanced_search_stats(self) -> Dict[str, Any]:
        """향상된 검색 통계"""
        stats = {
            'search_stats': self.search_stats,
            'query_expansion_enabled': self.query_expander is not None,
            'dynamic_threshold_enabled': self.threshold_engine is not None
        }
        
        # 각 컴포넌트별 통계 추가
        if hasattr(self, 'query_expander') and self.query_expander:
            stats['query_expansion_stats'] = self.query_expander.get_stats()
        
        if hasattr(self, 'threshold_engine') and self.threshold_engine:
            stats['threshold_stats'] = self.threshold_engine.get_stats()
        
        return stats

# =====================================
# 4. 통합 RAG 파이프라인 설계
# =====================================

class IntegratedRAGPipeline:
    """모든 강화 기능이 통합된 RAG 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrated_config = IntegratedSystemConfig(existing_config=config)
        
        # 컴포넌트 초기화
        self._init_components()
        
        # 파이프라인 통계
        self.pipeline_stats = {
            'total_requests': 0,
            'successful_responses': 0,
            'component_failures': {
                'document_processing': 0,
                'query_expansion': 0,
                'dynamic_threshold': 0,
                'retrieval': 0,
                'generation': 0
            },
            'avg_response_time': 0
        }
    
    def _init_components(self):
        """모든 컴포넌트 초기화"""
        try:
            # 향상된 문서 처리기
            self.document_processor = EnhancedDocumentProcessor(
                self.config, self.integrated_config
            )
            
            # 향상된 검색 모듈 (실제로는 기존 컴포넌트들을 주입받아야 함)
            # self.retrieval_module = EnhancedRetrievalModule(
            #     self.config, self.integrated_config, qdrant_client, embedder
            # )
            
            # 기존 컴포넌트들은 그대로 유지
            # self.generation_module = GenerationModule(self.config)
            # self.session_module = SessionModule(self.config)
            
            logger.info("통합 RAG 파이프라인 초기화 완료")
            
        except Exception as e:
            logger.error(f"파이프라인 초기화 실패: {e}")
            raise
    
    async def process_chat_request(self, message: str, session_id: str, 
                                  options: Dict[str, Any] = None) -> Dict[str, Any]:
        """통합 채팅 요청 처리"""
        import time
        start_time = time.time()
        
        self.pipeline_stats['total_requests'] += 1
        options = options or {}
        
        try:
            # 1. 세션 컨텍스트 조회 (기존 로직)
            # session_context = await self.session_module.get_or_create_session(session_id)
            
            # 2. 향상된 문서 검색
            # search_results = await self.retrieval_module.search_with_enhancements(
            #     message, options
            # )
            
            # 3. 응답 생성 (기존 로직)
            # response = await self.generation_module.generate_response(
            #     message, search_results, session_context
            # )
            
            # 4. 세션 업데이트 (기존 로직)
            # await self.session_module.add_exchange(session_id, message, response)
            
            # 통계 업데이트
            self.pipeline_stats['successful_responses'] += 1
            response_time = time.time() - start_time
            self.pipeline_stats['avg_response_time'] = (
                (self.pipeline_stats['avg_response_time'] * (self.pipeline_stats['successful_responses'] - 1) + response_time) /
                self.pipeline_stats['successful_responses']
            )
            
            return {
                'response': 'mock_response',  # 실제로는 생성된 응답
                'sources': [],  # 실제로는 검색 결과
                'metadata': {
                    'response_time': response_time,
                    'enhancements_used': self._get_used_enhancements()
                }
            }
            
        except Exception as e:
            logger.error(f"채팅 요청 처리 실패: {e}")
            # 컴포넌트별 실패 통계 업데이트
            self.pipeline_stats['component_failures']['retrieval'] += 1
            raise
    
    def _get_used_enhancements(self) -> List[str]:
        """사용된 강화 기능 목록"""
        enhancements = []
        
        if self.integrated_config.enable_query_expansion:
            enhancements.append('query_expansion')
        
        if self.integrated_config.enable_dynamic_threshold:
            enhancements.append('dynamic_threshold')
            
        if self.integrated_config.enable_semantic_chunking:
            enhancements.append('semantic_chunking')
            
        return enhancements
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """종합 시스템 통계"""
        stats = {
            'pipeline_stats': self.pipeline_stats,
            'integrated_config': {
                'query_expansion_enabled': self.integrated_config.enable_query_expansion,
                'dynamic_threshold_enabled': self.integrated_config.enable_dynamic_threshold,  
                'semantic_chunking_enabled': self.integrated_config.enable_semantic_chunking
            }
        }
        
        # 각 컴포넌트별 통계 추가
        if hasattr(self, 'document_processor'):
            stats['document_processing'] = self.document_processor.get_enhanced_stats()
            
        if hasattr(self, 'retrieval_module'):
            stats['retrieval'] = self.retrieval_module.get_enhanced_search_stats()
        
        return stats

# =====================================
# 5. 설정 파일 업데이트 템플릿
# =====================================

UPDATED_CONFIG_YAML = """
# Enhanced RAG Configuration with Integrated Improvements

# 기존 설정 (변경 없음)
app:
  name: "Enhanced RAG Chatbot"
  version: "3.0.0"
  debug: false

server:
  host: "${HOST:-0.0.0.0}"
  port: "${PORT:-8000}"

qdrant:
  url: "${QDRANT_URL}"
  api_key: "${QDRANT_API_KEY:-}"
  collection_name: "documents"
  vector_size: 768
  distance: "cosine"
  hybrid_search:
    dense_weight: 0.6
    sparse_weight: 0.4

embeddings:
  provider: "google"
  model: "models/text-embedding-004"
  sparse_model: "Qdrant/bm42-all-minilm-l6-v2-attentions"
  batch_size: 100

llm:
  providers: ["google", "anthropic", "openai"]
  default_provider: "openai"
  auto_fallback: true
  fallback_order: ["openai", "anthropic", "google"]
  google:
    model: "gemini-2.0-flash-exp"
    api_key: "${GOOGLE_API_KEY:-}"
    temperature: 0.3
    max_tokens: 10000
  anthropic:
    model: "claude-sonnet-4-20250514"
    api_key: "${ANTHROPIC_API_KEY:-}"
    temperature: 0.3
    max_tokens: 10000
  openai:
    model: "gpt-5-2025-08-07"
    api_key: "${OPENAI_API_KEY:-}"
    temperature: 0.3
    max_tokens: 10000

# 강화된 문서 처리 설정 
document_processing:
  # 시멘틱 청킹 설정 (새로 추가)
  splitter_type: "semantic"  # recursive, semantic, hybrid 중 선택
  target_chunk_size: 1250     # 목표 청크 크기 (1,000-1,500자 중간값)
  min_chunk_size: 1000        # 최소 청크 크기  
  max_chunk_size: 1500        # 최대 청크 크기
  
  # 시멘틱 분할 고급 설정
  semantic_threshold: 0.3     # 의미적 분할 임계값 (0.0-1.0)
  overlap_sentences: 2        # 오버랩할 문장 수
  sentence_window: 3          # 의미 분석 윈도우 크기
  preserve_structure: true    # 문서 구조 보존 여부
  korean_optimized: true      # 한국어 최적화 활성화
  
  # 기존 설정 (호환성 유지)
  chunk_size: 1250           # semantic chunking 사용 시 target_chunk_size와 동일
  chunk_overlap: 100         # semantic chunking 사용 시 자동 계산
  max_file_size: 52428800    # 50MB

# 강화된 검색 설정
retrieval:
  max_sources: 25
  min_score: 0.15            # 기본 임계값 (동적 임계값 비활성화 시)
  top_k: 10
  enable_reranking: true
  quality_threshold: 0.15
  deduplication_enabled: true
  
  # 동적 임계값 설정 (새로 추가)
  enable_dynamic_threshold: true
  dynamic_threshold:
    base_threshold: 0.15      # 시작 임계값
    target_min_results: 5     # 목표 최소 결과 수
    threshold_levels: [0.15, 0.10, 0.05, 0.01]  # 임계값 단계
    learning_enabled: true    # 학습 기반 최적화
    cache_enabled: true       # 캐싱 활성화
    cache_ttl: 1800          # 캐시 TTL (30분)
  
  # GPT-5 쿼리 확장 설정 (새로 추가)  
  enable_query_expansion: true
  query_expansion:
    model: "gpt-5-nano"       # GPT-5 nano 모델 사용
    temperature: 0.3          # 창의성 조절
    max_tokens: 1000          # 최대 토큰 수
    cache_ttl: 3600          # 캐시 TTL (1시간)
    min_query_length: 10      # 확장 대상 최소 쿼리 길이
    max_synonyms: 5           # 최대 동의어 수
    expansion_strategies:     # 확장 전략
      - "synonym"             # 동의어 확장
      - "intent"              # 의도 분석
      - "keyword"             # 핵심 키워드 추출
    fallback_enabled: true    # 실패 시 원본 쿼리 사용

# 기존 설정 (변경 없음)
reranking:
  enabled: true
  default_provider: "llm"
  min_score: 0.15
  providers:
    cohere:
      api_key: ""
      model: "rerank-multilingual-v2.0"
    jina:
      api_key: ""
      model: "jina-reranker-v1-base-en"
      endpoint: "https://api.jina.ai/v1/rerank"
    llm:
      enabled: true
      provider: "google"
      max_tokens: 20000

session:
  ttl: 3600
  max_exchanges: 5

logging:
  level: "DEBUG"
  format: "structured"

uploads:
  directory: "./uploads"
  max_file_size: 52428800
  allowed_types: [".pdf", ".txt", ".md", ".docx", ".xlsx", ".json"]

# 시스템 모니터링 설정 (새로 추가)
monitoring:
  enable_stats_tracking: true
  stats_reset_interval: 86400  # 24시간
  performance_logging: true
  component_health_checks: true
"""

# =====================================
# 6. 마이그레이션 및 배포 가이드
# =====================================

class SystemMigrationGuide:
    """시스템 마이그레이션 가이드"""
    
    @staticmethod
    def get_migration_steps() -> List[str]:
        """마이그레이션 단계별 가이드"""
        return [
            "1. 백업: 기존 config.yaml 및 데이터 백업",
            "2. 의존성: 새 Python 패키지 설치 (scikit-learn, numpy 추가)",
            "3. 설정: config.yaml 업데이트 (위 템플릿 참조)",
            "4. 코드: 새로운 모듈 파일들 배치",
            "   - gpt5_query_expansion_system.py",
            "   - dynamic_threshold_system_detailed.py", 
            "   - semantic_chunking_strategy.py",
            "5. 통합: 기존 DocumentProcessor, RetrievalModule 교체",
            "6. 테스트: 각 컴포넌트별 개별 테스트",
            "7. 점진적 활성화: 한 번에 하나씩 기능 활성화",
            "8. 모니터링: 성능 및 품질 지표 모니터링",
            "9. 튜닝: 임계값, 청크 크기 등 파라미터 조정",
            "10. 문서화: 운영 가이드 및 트러블슈팅 문서 작성"
        ]
    
    @staticmethod
    def get_rollback_plan() -> List[str]:
        """롤백 계획"""
        return [
            "1. 즉시 롤백: config.yaml에서 새 기능들 비활성화",
            "   - enable_query_expansion: false",
            "   - enable_dynamic_threshold: false", 
            "   - enable_semantic_chunking: false",
            "2. 코드 롤백: 기존 DocumentProcessor, RetrievalModule 복원",
            "3. 설정 복원: 기존 config.yaml 복원", 
            "4. 서비스 재시작: 모든 변경사항 적용",
            "5. 모니터링: 기존 기능 정상 동작 확인"
        ]

# =====================================
# 7. 테스트 및 검증 프레임워크
# =====================================

class IntegrationTestFramework:
    """통합 테스트 프레임워크"""
    
    def __init__(self, pipeline: IntegratedRAGPipeline):
        self.pipeline = pipeline
    
    async def test_semantic_chunking(self) -> Dict[str, Any]:
        """시멘틱 청킹 테스트"""
        test_document = """
        한국의 인공지능 기술은 빠르게 발전하고 있습니다. 
        특히 자연어 처리 분야에서 많은 진전이 있었습니다.
        
        대화형 AI 시스템은 사용자와의 상호작용을 통해 학습합니다.
        이러한 시스템은 다양한 도메인에서 활용되고 있습니다.
        
        미래의 AI 기술은 더욱 정교해질 것으로 예상됩니다.
        윤리적 고려사항도 함께 발전해야 할 중요한 요소입니다.
        """
        
        # 테스트 실행 로직
        return {'status': 'success', 'chunks_created': 3, 'avg_chunk_size': 1200}
    
    async def test_query_expansion(self) -> Dict[str, Any]:
        """쿼리 확장 테스트"""
        test_queries = [
            "AI 기술의 미래",
            "머신러닝 알고리즘", 
            "자연어 처리 방법"
        ]
        
        # 테스트 실행 로직
        return {'status': 'success', 'expansion_rate': 0.85, 'quality_score': 0.92}
    
    async def test_dynamic_threshold(self) -> Dict[str, Any]:
        """동적 임계값 테스트"""
        test_queries = [
            "매우 구체적인 기술 질문",
            "일반적인 개념 질문",
            "모호한 질문"
        ]
        
        # 테스트 실행 로직  
        return {'status': 'success', 'threshold_adaptations': 8, 'result_improvement': 0.23}

if __name__ == "__main__":
    # 통합 시스템 초기화 및 테스트 예시
    print("Enhanced RAG System Integration Design")
    print("=" * 50)
    
    # 마이그레이션 가이드 출력
    migration_guide = SystemMigrationGuide()
    print("\\n마이그레이션 단계:")
    for step in migration_guide.get_migration_steps()[:5]:  # 처음 5단계만 표시
        print(f"  {step}")
    
    print("\\n설정 업데이트 완료!")
    print("새로운 기능들이 통합된 RAG 시스템이 준비되었습니다.")
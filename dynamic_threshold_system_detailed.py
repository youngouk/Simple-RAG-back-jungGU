"""
동적 임계값 시스템 초상세 설계
Ultra-Detailed Dynamic Threshold System for RAG Search

핵심 목표:
1. 결과가 5개 미만일 때 자동 임계값 완화
2. 일관된 임계값 관리 (현재 0.15/0.5/0.01 문제 해결)
3. 성능 모니터링 및 최적화
4. MVP 환경에 최적화된 간단함
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ThresholdLevel(Enum):
    """임계값 레벨 정의"""
    STRICT = "strict"      # 0.15 - 엄격한 기준
    MODERATE = "moderate"  # 0.10 - 중간 기준  
    RELAXED = "relaxed"    # 0.05 - 완화된 기준
    MINIMAL = "minimal"    # 0.01 - 최소 기준

class SearchContext(Enum):
    """검색 맥락 분류"""
    FACTUAL = "factual"           # 사실 검색
    SEMANTIC = "semantic"         # 의미적 검색
    KEYWORD = "keyword"          # 키워드 매칭
    CONVERSATIONAL = "conversational"  # 대화형 검색

@dataclass
class ThresholdConfig:
    """동적 임계값 설정"""
    # 기본 임계값 레벨별 설정
    strict_threshold: float = 0.15      # 엄격한 기준 (고품질 결과)
    moderate_threshold: float = 0.10    # 중간 기준 (균형)
    relaxed_threshold: float = 0.05     # 완화된 기준 (포괄적)
    minimal_threshold: float = 0.01     # 최소 기준 (모든 가능한 결과)
    
    # 동적 조정 기준
    target_min_results: int = 5         # 최소 목표 결과 수
    ideal_result_range: Tuple[int, int] = (8, 12)  # 이상적 결과 수 범위
    max_adjustment_attempts: int = 3     # 최대 조정 시도 횟수
    
    # 성능 최적화
    enable_caching: bool = True         # 임계값 결과 캐싱
    cache_ttl: int = 300               # 캐시 TTL (5분)
    enable_learning: bool = True        # 학습 기반 최적화
    
    # 맥락별 기본값
    context_defaults: Dict[str, float] = None
    
    def __post_init__(self):
        if self.context_defaults is None:
            self.context_defaults = {
                SearchContext.FACTUAL.value: self.strict_threshold,
                SearchContext.SEMANTIC.value: self.moderate_threshold,
                SearchContext.KEYWORD.value: self.relaxed_threshold,
                SearchContext.CONVERSATIONAL.value: self.moderate_threshold
            }

@dataclass
class SearchAttempt:
    """검색 시도 기록"""
    threshold: float
    result_count: int
    response_time: float
    query_hash: str
    context: SearchContext
    timestamp: float
    success: bool
    quality_score: Optional[float] = None

class ThresholdCache:
    """임계값 결과 캐싱"""
    
    def __init__(self, ttl: int = 300):
        self.cache: Dict[str, Tuple[float, int, float]] = {}  # query_hash -> (threshold, count, timestamp)
        self.ttl = ttl
    
    def get(self, query_hash: str) -> Optional[Tuple[float, int]]:
        """캐시에서 임계값 조회"""
        if query_hash in self.cache:
            threshold, count, timestamp = self.cache[query_hash]
            if time.time() - timestamp < self.ttl:
                return threshold, count
            else:
                del self.cache[query_hash]
        return None
    
    def set(self, query_hash: str, threshold: float, result_count: int):
        """캐시에 임계값 저장"""
        self.cache[query_hash] = (threshold, result_count, time.time())
    
    def clear_expired(self):
        """만료된 캐시 항목 제거"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, _, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

class DynamicThresholdEngine:
    """동적 임계값 엔진"""
    
    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.cache = ThresholdCache(config.cache_ttl) if config.enable_caching else None
        
        # 성능 통계
        self.stats = {
            'total_searches': 0,
            'successful_adjustments': 0,
            'threshold_distributions': defaultdict(int),
            'context_performance': defaultdict(list),
            'average_attempts': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'quality_improvements': 0
        }
        
        # 학습 시스템
        self.learning_history: deque = deque(maxlen=1000)  # 최근 1000회 기록
        self.context_patterns: Dict[SearchContext, List[float]] = defaultdict(list)
        
        # 임계값 레벨 매핑
        self.threshold_levels = {
            ThresholdLevel.STRICT: config.strict_threshold,
            ThresholdLevel.MODERATE: config.moderate_threshold,
            ThresholdLevel.RELAXED: config.relaxed_threshold,
            ThresholdLevel.MINIMAL: config.minimal_threshold
        }
    
    async def find_optimal_threshold(
        self,
        query: str,
        search_function: Callable,
        context: SearchContext = SearchContext.SEMANTIC,
        **search_kwargs
    ) -> Tuple[float, List[Any], Dict[str, Any]]:
        """
        최적 임계값을 찾아 검색 실행
        
        Returns:
            (사용된_임계값, 검색_결과, 메타데이터)
        """
        start_time = time.time()
        self.stats['total_searches'] += 1
        query_hash = str(hash(query))
        
        # 1. 캐시 확인
        if self.cache:
            cached_result = self.cache.get(query_hash)
            if cached_result:
                threshold, expected_count = cached_result
                self.stats['cache_hits'] += 1
                logger.debug(f"캐시 히트: 임계값={threshold:.3f}, 예상결과={expected_count}")
                
                # 캐시된 임계값으로 직접 검색
                search_kwargs['min_score'] = threshold
                results = await search_function(query, **search_kwargs)
                
                return threshold, results, {
                    'cache_hit': True,
                    'attempts': 1,
                    'processing_time': time.time() - start_time
                }
            else:
                self.stats['cache_misses'] += 1
        
        # 2. 학습 기반 시작점 결정
        starting_threshold = self._determine_starting_threshold(query, context)
        logger.debug(f"시작 임계값: {starting_threshold:.3f} (맥락: {context.value})")
        
        # 3. 단계적 임계값 조정
        attempts_data = []
        best_threshold = starting_threshold
        best_results = []
        
        # 임계값 조정 순서 결정
        adjustment_sequence = self._create_adjustment_sequence(starting_threshold)
        
        for attempt, threshold in enumerate(adjustment_sequence, 1):
            logger.debug(f"검색 시도 {attempt}/{len(adjustment_sequence)}: 임계값={threshold:.3f}")
            
            # 검색 실행
            search_kwargs['min_score'] = threshold
            attempt_start = time.time()
            
            try:
                results = await search_function(query, **search_kwargs)
                response_time = time.time() - attempt_start
                
                # 시도 기록
                search_attempt = SearchAttempt(
                    threshold=threshold,
                    result_count=len(results),
                    response_time=response_time,
                    query_hash=query_hash,
                    context=context,
                    timestamp=time.time(),
                    success=True
                )
                attempts_data.append(search_attempt)
                
                logger.info(f"임계값 {threshold:.3f}: {len(results)}개 결과 ({response_time:.3f}s)")
                
                # 결과 평가
                if self._is_satisfactory_result(len(results), threshold, context):
                    best_threshold = threshold
                    best_results = results
                    self.stats['successful_adjustments'] += 1
                    break
                
                # 최선의 결과 업데이트 (결과가 있는 경우)
                if len(results) > len(best_results):
                    best_threshold = threshold
                    best_results = results
                    
            except Exception as e:
                logger.error(f"임계값 {threshold:.3f} 검색 실패: {e}")
                search_attempt = SearchAttempt(
                    threshold=threshold,
                    result_count=0,
                    response_time=time.time() - attempt_start,
                    query_hash=query_hash,
                    context=context,
                    timestamp=time.time(),
                    success=False
                )
                attempts_data.append(search_attempt)
        
        # 4. 결과 후처리 및 학습
        total_time = time.time() - start_time
        metadata = self._create_result_metadata(attempts_data, total_time, context)
        
        # 학습 데이터 업데이트
        if self.config.enable_learning and best_results:
            self._update_learning_data(attempts_data, context)
        
        # 캐시 업데이트
        if self.cache and best_results:
            self.cache.set(query_hash, best_threshold, len(best_results))
        
        # 통계 업데이트
        self._update_stats(attempts_data, context)
        
        logger.info(
            f"동적 임계값 완료: {best_threshold:.3f} → {len(best_results)}개 결과 "
            f"({len(attempts_data)}회 시도, {total_time:.3f}s)"
        )
        
        return best_threshold, best_results, metadata
    
    def _determine_starting_threshold(self, query: str, context: SearchContext) -> float:
        """학습 기반 시작 임계값 결정"""
        # 1. 맥락별 기본값
        base_threshold = self.config.context_defaults[context.value]
        
        # 2. 학습 기반 조정
        if self.config.enable_learning and self.context_patterns[context]:
            # 해당 맥락에서의 평균 성공 임계값
            recent_successful = [
                attempt.threshold for attempt in self.learning_history
                if attempt.context == context and attempt.success and attempt.result_count >= self.config.target_min_results
            ]
            
            if recent_successful:
                learned_threshold = statistics.median(recent_successful)
                # 학습된 값과 기본값의 가중 평균 (7:3 비율)
                base_threshold = learned_threshold * 0.7 + base_threshold * 0.3
                logger.debug(f"학습 조정된 시작 임계값: {base_threshold:.3f}")
        
        return base_threshold
    
    def _create_adjustment_sequence(self, starting_threshold: float) -> List[float]:
        """임계값 조정 순서 생성"""
        all_thresholds = list(self.threshold_levels.values())
        
        # 시작점에서 가장 가까운 순서로 정렬
        sorted_thresholds = sorted(
            all_thresholds,
            key=lambda x: abs(x - starting_threshold)
        )
        
        # 시작 임계값을 첫 번째로, 나머지를 내림차순으로
        sequence = [starting_threshold]
        for threshold in sorted(all_thresholds, reverse=True):
            if threshold != starting_threshold:
                sequence.append(threshold)
        
        return sequence[:self.config.max_adjustment_attempts]
    
    def _is_satisfactory_result(
        self, 
        result_count: int, 
        threshold: float, 
        context: SearchContext
    ) -> bool:
        """결과 만족도 평가"""
        # 1. 최소 결과 수 확인
        if result_count < self.config.target_min_results:
            return False
        
        # 2. 이상적 범위 확인
        min_ideal, max_ideal = self.config.ideal_result_range
        if min_ideal <= result_count <= max_ideal:
            return True
        
        # 3. 맥락별 기준
        if context == SearchContext.FACTUAL:
            # 사실 검색은 정확성 우선
            return result_count >= self.config.target_min_results and threshold >= self.config.moderate_threshold
        elif context == SearchContext.KEYWORD:
            # 키워드 검색은 포괄성 우선
            return result_count >= self.config.target_min_results
        
        # 4. 기본: 최소 조건 만족
        return result_count >= self.config.target_min_results
    
    def _create_result_metadata(
        self, 
        attempts: List[SearchAttempt], 
        total_time: float,
        context: SearchContext
    ) -> Dict[str, Any]:
        """결과 메타데이터 생성"""
        return {
            'total_attempts': len(attempts),
            'successful_attempts': sum(1 for a in attempts if a.success),
            'total_processing_time': total_time,
            'average_response_time': statistics.mean([a.response_time for a in attempts]) if attempts else 0.0,
            'thresholds_tried': [a.threshold for a in attempts],
            'result_counts': [a.result_count for a in attempts],
            'context': context.value,
            'cache_hit': False,
            'learning_applied': self.config.enable_learning and bool(self.context_patterns[context])
        }
    
    def _update_learning_data(self, attempts: List[SearchAttempt], context: SearchContext):
        """학습 데이터 업데이트"""
        for attempt in attempts:
            self.learning_history.append(attempt)
            
            if attempt.success:
                self.context_patterns[context].append(attempt.threshold)
                
                # 컨텍스트별 패턴 제한 (최근 100개만)
                if len(self.context_patterns[context]) > 100:
                    self.context_patterns[context] = self.context_patterns[context][-100:]
    
    def _update_stats(self, attempts: List[SearchAttempt], context: SearchContext):
        """통계 업데이트"""
        for attempt in attempts:
            self.stats['threshold_distributions'][f"{attempt.threshold:.2f}"] += 1
            self.stats['context_performance'][context.value].append(attempt.result_count)
        
        # 평균 시도 횟수 업데이트
        total_searches = self.stats['total_searches']
        current_avg = self.stats['average_attempts']
        self.stats['average_attempts'] = (
            (current_avg * (total_searches - 1) + len(attempts)) / total_searches
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        # 컨텍스트별 평균 결과 수
        context_averages = {}
        for context, results in self.stats['context_performance'].items():
            if results:
                context_averages[context] = {
                    'average_results': statistics.mean(results),
                    'median_results': statistics.median(results),
                    'total_searches': len(results)
                }
        
        return {
            **self.stats,
            'context_averages': context_averages,
            'success_rate': (
                self.stats['successful_adjustments'] / max(1, self.stats['total_searches'])
            ) * 100,
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            ) * 100 if self.cache else 0,
            'learning_data_size': len(self.learning_history),
            'context_patterns_size': {
                context.value: len(patterns) 
                for context, patterns in self.context_patterns.items()
            }
        }
    
    def optimize_thresholds(self):
        """임계값 최적화 (주기적 실행 권장)"""
        if not self.config.enable_learning or not self.learning_history:
            return
        
        logger.info("임계값 학습 기반 최적화 시작")
        
        # 컨텍스트별 최적 임계값 계산
        for context in SearchContext:
            successful_attempts = [
                attempt for attempt in self.learning_history
                if attempt.context == context and 
                   attempt.success and 
                   attempt.result_count >= self.config.target_min_results
            ]
            
            if len(successful_attempts) >= 5:  # 최소 5개 데이터 필요
                optimal_threshold = statistics.median([a.threshold for a in successful_attempts])
                current_default = self.config.context_defaults[context.value]
                
                # 10% 이상 차이나면 업데이트
                if abs(optimal_threshold - current_default) > 0.015:
                    logger.info(
                        f"컨텍스트 {context.value} 임계값 최적화: "
                        f"{current_default:.3f} → {optimal_threshold:.3f}"
                    )
                    self.config.context_defaults[context.value] = optimal_threshold
        
        # 캐시 정리
        if self.cache:
            self.cache.clear_expired()

# 기존 RetrievalModule과의 통합
class EnhancedRetrievalWithDynamicThreshold:
    """동적 임계값이 적용된 검색 모듈"""
    
    def __init__(self, original_retrieval_module, config: Dict[str, Any]):
        self.original_module = original_retrieval_module
        
        # 동적 임계값 엔진 초기화
        threshold_config = ThresholdConfig(
            strict_threshold=config.get('retrieval', {}).get('min_score', 0.15),
            target_min_results=config.get('retrieval', {}).get('target_min_results', 5),
            enable_caching=config.get('dynamic_threshold', {}).get('enable_caching', True),
            enable_learning=config.get('dynamic_threshold', {}).get('enable_learning', True)
        )
        
        self.threshold_engine = DynamicThresholdEngine(threshold_config)
        
        # 주기적 최적화 (10분마다)
        self._optimization_task = None
        self._start_optimization_loop()
    
    async def search(self, query: str, options: Dict[str, Any] = None) -> List[Any]:
        """동적 임계값을 사용한 검색"""
        options = options or {}
        
        # 동적 임계값 비활성화 옵션
        if options.get('disable_dynamic_threshold', False):
            return await self.original_module.search(query, options)
        
        # 검색 맥락 결정
        context = self._determine_search_context(query, options)
        
        # 동적 임계값 검색
        threshold, results, metadata = await self.threshold_engine.find_optimal_threshold(
            query=query,
            search_function=self._search_wrapper,
            context=context,
            **options
        )
        
        # 결과에 메타데이터 추가
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata.update({
                    'dynamic_threshold': threshold,
                    'threshold_metadata': metadata
                })
            elif isinstance(result, dict):
                result.update({
                    'dynamic_threshold': threshold,
                    'threshold_metadata': metadata
                })
        
        return results
    
    async def _search_wrapper(self, query: str, **kwargs) -> List[Any]:
        """기존 검색 모듈 래퍼"""
        # min_score 외의 모든 옵션 전달
        search_options = {k: v for k, v in kwargs.items() if k != 'min_score'}
        search_options['min_score'] = kwargs.get('min_score', 0.15)
        
        return await self.original_module.search(query, search_options)
    
    def _determine_search_context(self, query: str, options: Dict[str, Any]) -> SearchContext:
        """검색 맥락 결정 (간단한 휴리스틱)"""
        query_lower = query.lower()
        
        # 질문 형태
        if any(word in query_lower for word in ['무엇', '어떻게', '왜', '언제', '어디서']):
            return SearchContext.CONVERSATIONAL
        
        # 키워드성 검색
        if len(query.split()) <= 2:
            return SearchContext.KEYWORD
        
        # 기본값: 의미적 검색
        return SearchContext.SEMANTIC
    
    def _start_optimization_loop(self):
        """최적화 루프 시작"""
        async def optimization_loop():
            while True:
                try:
                    await asyncio.sleep(600)  # 10분 대기
                    self.threshold_engine.optimize_thresholds()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"임계값 최적화 오류: {e}")
        
        self._optimization_task = asyncio.create_task(optimization_loop())
    
    async def close(self):
        """리소스 정리"""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
    
    def get_threshold_stats(self) -> Dict[str, Any]:
        """임계값 통계 조회"""
        return self.threshold_engine.get_performance_stats()

# 설정 업데이트 예시
enhanced_config_yaml = """
# 동적 임계값 설정
dynamic_threshold:
  enabled: true
  target_min_results: 5
  enable_caching: true
  cache_ttl: 300
  enable_learning: true
  
  # 임계값 레벨별 설정
  thresholds:
    strict: 0.15      # 고품질 결과
    moderate: 0.10    # 균형 잡힌 결과  
    relaxed: 0.05     # 포괄적 결과
    minimal: 0.01     # 모든 가능한 결과
  
  # 맥락별 기본값
  context_defaults:
    factual: 0.15
    semantic: 0.10
    keyword: 0.05
    conversational: 0.10

# 기존 retrieval 설정 업데이트
retrieval:
  max_sources: 25
  min_score: 0.15  # 동적 임계값의 시작점으로 사용
  target_min_results: 5
  enable_reranking: true
"""

# 사용 예시
async def example_dynamic_threshold():
    """동적 임계값 사용 예시"""
    config = {
        'retrieval': {'min_score': 0.15, 'target_min_results': 5},
        'dynamic_threshold': {'enable_caching': True, 'enable_learning': True}
    }
    
    # 기존 검색 모듈과 통합 (의사 코드)
    # original_retrieval = RetrievalModule(config)
    # enhanced_retrieval = EnhancedRetrievalWithDynamicThreshold(original_retrieval, config)
    
    # 검색 실행
    # results = await enhanced_retrieval.search("머신러닝 알고리즘")
    # stats = enhanced_retrieval.get_threshold_stats()
    
    pass

if __name__ == "__main__":
    asyncio.run(example_dynamic_threshold())
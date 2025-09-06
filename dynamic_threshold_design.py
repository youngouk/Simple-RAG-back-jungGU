# 동적 임계값 시스템 설계
# Dynamic Threshold System Design

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ThresholdConfig:
    """동적 임계값 설정"""
    base_threshold: float = 0.15        # 기본 임계값 15%
    target_min_results: int = 5         # 최소 목표 결과 수
    adjustment_steps: List[float] = None # 조정 단계
    max_attempts: int = 3               # 최대 시도 횟수
    
    def __post_init__(self):
        if self.adjustment_steps is None:
            self.adjustment_steps = [0.15, 0.10, 0.05, 0.01]

class DynamicThresholdManager:
    """동적 임계값 관리자"""
    
    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.stats = {
            'total_searches': 0,
            'threshold_adjustments': 0,
            'adjustment_success_rate': 0.0,
            'step_usage': {step: 0 for step in config.adjustment_steps}
        }
    
    async def get_optimal_threshold(
        self, 
        query: str,
        initial_search_func,
        **search_kwargs
    ) -> Tuple[float, List[Any]]:
        """
        최적 임계값을 찾아 검색 실행
        
        Args:
            query: 검색 쿼리
            initial_search_func: 검색 함수
            **search_kwargs: 검색 추가 인자
            
        Returns:
            (사용된_임계값, 검색_결과)
        """
        self.stats['total_searches'] += 1
        
        for attempt, threshold in enumerate(self.config.adjustment_steps):
            logger.debug(
                f"검색 시도 {attempt + 1}/{len(self.config.adjustment_steps)}: "
                f"임계값={threshold:.3f}"
            )
            
            # 검색 실행
            search_kwargs['min_score'] = threshold
            results = await initial_search_func(query, **search_kwargs)
            
            # 통계 업데이트
            self.stats['step_usage'][threshold] += 1
            
            # 결과 평가
            result_count = len(results)
            logger.info(
                f"임계값 {threshold:.3f}로 {result_count}개 결과 발견"
            )
            
            # 충분한 결과가 있으면 성공
            if result_count >= self.config.target_min_results:
                if attempt > 0:  # 조정이 있었다면
                    self.stats['threshold_adjustments'] += 1
                    self._update_success_rate()
                
                logger.info(
                    f"동적 임계값 성공: {threshold:.3f} "
                    f"(결과: {result_count}개)"
                )
                return threshold, results
            
            # 마지막 시도라면 결과를 반환
            if attempt == len(self.config.adjustment_steps) - 1:
                logger.warning(
                    f"모든 임계값 시도 완료: 최종 {result_count}개 결과"
                )
                if attempt > 0:
                    self.stats['threshold_adjustments'] += 1
                    self._update_success_rate()
                return threshold, results
        
        # 이론상 도달하지 않는 코드
        return self.config.adjustment_steps[-1], []
    
    def _update_success_rate(self):
        """성공률 업데이트"""
        if self.stats['threshold_adjustments'] > 0:
            # 임계값 조정이 있었던 경우 중 목표 달성 비율
            # 실제 구현에서는 더 정교한 성공률 계산 가능
            pass
    
    def get_threshold_stats(self) -> Dict[str, Any]:
        """임계값 조정 통계"""
        return {
            **self.stats,
            'config': {
                'base_threshold': self.config.base_threshold,
                'target_min_results': self.config.target_min_results,
                'adjustment_steps': self.config.adjustment_steps
            }
        }
    
    def should_adjust_threshold(self, result_count: int) -> bool:
        """임계값 조정 필요 여부"""
        return result_count < self.config.target_min_results

# 기존 RetrievalModule 통합 방안
class EnhancedRetrievalModule:
    """동적 임계값이 적용된 검색 모듈"""
    
    def __init__(self, config: Dict[str, Any], embedder):
        # 기존 초기화...
        self.threshold_manager = DynamicThresholdManager(
            ThresholdConfig(
                base_threshold=config.get('retrieval', {}).get('min_score', 0.15),
                target_min_results=config.get('retrieval', {}).get('target_min_results', 5)
            )
        )
    
    async def search_with_dynamic_threshold(
        self, 
        query: str, 
        options: Dict[str, Any] = None
    ) -> List[Any]:
        """동적 임계값을 사용한 검색"""
        options = options or {}
        
        # 동적 임계값 비활성화 옵션
        if options.get('disable_dynamic_threshold', False):
            return await self.original_search(query, options)
        
        # 동적 임계값으로 검색
        used_threshold, results = await self.threshold_manager.get_optimal_threshold(
            query=query,
            initial_search_func=self._search_with_threshold,
            limit=options.get('limit', 25)
        )
        
        # 메타데이터에 사용된 임계값 추가
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata['used_threshold'] = used_threshold
            elif isinstance(result, dict):
                result['used_threshold'] = used_threshold
        
        logger.info(
            f"동적 임계값 검색 완료: {used_threshold:.3f} "
            f"→ {len(results)}개 결과"
        )
        
        return results
    
    async def _search_with_threshold(
        self, 
        query: str, 
        min_score: float,
        limit: int = 25
    ) -> List[Any]:
        """특정 임계값으로 검색 (내부 메서드)"""
        # 기존 검색 로직을 임계값과 함께 호출
        options = {'min_score': min_score, 'limit': limit}
        return await self.original_search(query, options)

# 설정 파일 업데이트 예시 (config.yaml)
config_update_example = """
retrieval:
  max_sources: 25
  min_score: 0.15  # 기본 임계값
  target_min_results: 5  # 최소 목표 결과 수 (새 옵션)
  dynamic_threshold:
    enabled: true  # 동적 임계값 활성화
    adjustment_steps: [0.15, 0.10, 0.05, 0.01]  # 조정 단계
    max_attempts: 3
  enable_reranking: true
  quality_threshold: 0.15
"""

# 사용 예시
async def example_usage():
    """사용 예시"""
    config = {
        'retrieval': {
            'min_score': 0.15,
            'target_min_results': 5
        }
    }
    
    # 검색 모듈 초기화 (기존 임베더 필요)
    # retrieval_module = EnhancedRetrievalModule(config, embedder)
    
    # 동적 임계값 검색
    # results = await retrieval_module.search_with_dynamic_threshold(
    #     query="사용자 쿼리",
    #     options={'limit': 25}
    # )
    
    pass

# 로깅 예시
"""
2025-01-06 10:30:15 - INFO - 검색 시도 1/3: 임계값=0.150
2025-01-06 10:30:16 - INFO - 임계값 0.150로 3개 결과 발견
2025-01-06 10:30:16 - DEBUG - 검색 시도 2/3: 임계값=0.100  
2025-01-06 10:30:17 - INFO - 임계값 0.100로 7개 결과 발견
2025-01-06 10:30:17 - INFO - 동적 임계값 성공: 0.100 (결과: 7개)
"""
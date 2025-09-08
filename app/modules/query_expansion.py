"""
GPT-5-nano 기반 지능형 쿼리 확장 시스템
Ultra-detailed Design for Query Expansion using GPT-5-nano
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """쿼리 복잡도 분류"""
    SIMPLE = "simple"       # 간단한 키워드 검색
    MEDIUM = "medium"       # 일반적인 질문
    COMPLEX = "complex"     # 복합적/추상적 질문
    CONTEXTUAL = "contextual"  # 맥락 의존적 질문

class SearchIntent(Enum):
    """검색 의도 분류"""
    FACTUAL = "factual"           # 사실 정보 요청
    PROCEDURAL = "procedural"     # 절차/방법 질문
    CONCEPTUAL = "conceptual"     # 개념 설명 요청
    COMPARATIVE = "comparative"   # 비교/분석 요청
    PROBLEM_SOLVING = "problem_solving"  # 문제 해결

@dataclass
class ExpandedQuery:
    """확장된 쿼리 구조"""
    original_query: str
    synonyms: List[str]
    related_terms: List[str]
    core_keywords: List[Dict[str, Any]]  # {"keyword": str, "weight": float}
    intent: SearchIntent
    complexity: QueryComplexity
    expanded_queries: List[Dict[str, Any]]  # {"query": str, "weight": float, "focus": str}
    search_strategy: str

class GPT5QueryExpansionEngine:
    """GPT-5-nano 기반 쿼리 확장 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_config = config.get('llm', {}).get('openai', {})
        
        # GPT-5-nano 클라이언트 초기화
        self.client = OpenAI(
            api_key=self.openai_config.get('api_key')
        )
        
        # 성능 통계
        self.stats = {
            'total_expansions': 0,
            'successful_expansions': 0,
            'average_response_time': 0.0,
            'complexity_distribution': {c.value: 0 for c in QueryComplexity},
            'intent_distribution': {i.value: 0 for i in SearchIntent},
            'gpt5_api_calls': 0,
            'json_parse_failures': 0
        }
        
        # 최적화된 프롬프트 템플릿
        self.expansion_prompt = self._create_expansion_prompt()
    
    def _create_expansion_prompt(self) -> str:
        """GPT-5-nano용 최적화된 쿼리 확장 프롬프트"""
        return """당신은 한국어 문서 검색을 위한 쿼리 확장 전문가입니다.

주어진 사용자 쿼리를 분석하고 검색 효율성을 극대화하기 위해 확장해주세요.

**분석 요구사항:**
1. 동의어 및 유사어 발굴 (한국어 특성 고려)
2. 핵심 키워드 추출 및 중요도 가중치 부여 (0.1-1.0)
3. 검색 의도 분류 (factual/procedural/conceptual/comparative/problem_solving)
4. 쿼리 복잡도 평가 (simple/medium/complex/contextual)
5. 다양한 관점의 확장 쿼리 생성 (각 쿼리별 가중치 포함)

**응답 형식:** 반드시 아래 JSON 구조를 정확히 따라주세요.

```json
{
  "original_query": "원본 쿼리",
  "synonyms": ["동의어1", "동의어2", "동의어3"],
  "related_terms": ["관련용어1", "관련용어2", "관련용어3"],
  "core_keywords": [
    {"keyword": "핵심키워드1", "weight": 0.9},
    {"keyword": "핵심키워드2", "weight": 0.7},
    {"keyword": "핵심키워드3", "weight": 0.5}
  ],
  "intent": "factual|procedural|conceptual|comparative|problem_solving",
  "complexity": "simple|medium|complex|contextual",
  "expanded_queries": [
    {"query": "확장쿼리1", "weight": 1.0, "focus": "주요_관점"},
    {"query": "확장쿼리2", "weight": 0.8, "focus": "보조_관점"},
    {"query": "확장쿼리3", "weight": 0.6, "focus": "세부_관점"}
  ],
  "search_strategy": "broad|focused|hybrid|contextual"
}
```

**사용자 쿼리:** {query}

**지시사항:**
- JSON 형식을 정확히 준수하세요
- weight는 0.1-1.0 범위의 실수여야 합니다
- 한국어 언어적 특성을 고려하세요
- 검색 효율성에 집중하세요"""

    async def expand_query(self, query: str) -> Optional[ExpandedQuery]:
        """
        쿼리 확장 메인 메서드
        
        Args:
            query: 원본 사용자 쿼리
            
        Returns:
            ExpandedQuery 객체 또는 None (실패시)
        """
        start_time = asyncio.get_event_loop().time()
        self.stats['total_expansions'] += 1
        
        try:
            # 1. 사전 필터링 - 간단한 쿼리는 빠른 처리
            if self._is_simple_query(query):
                logger.info(f"간단한 쿼리로 분류: {query[:50]}...")
                return self._create_simple_expansion(query)
            
            # 2. GPT-5-nano 호출
            logger.debug(f"GPT-5-nano 쿼리 확장 시작: {query}")
            expanded_data = await self._call_gpt5_nano(query)
            
            if not expanded_data:
                logger.warning(f"GPT-5 확장 실패, 폴백 사용: {query}")
                return self._create_fallback_expansion(query)
            
            # 3. 구조화된 객체 생성
            expanded_query = self._parse_expansion_result(expanded_data, query)
            
            # 4. 통계 업데이트
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_stats(expanded_query, processing_time)
            
            logger.info(
                f"쿼리 확장 완료: {len(expanded_query.expanded_queries)}개 확장, "
                f"의도={expanded_query.intent.value}, "
                f"복잡도={expanded_query.complexity.value}"
            )
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"쿼리 확장 오류: {e}")
            return self._create_fallback_expansion(query)
    
    def _is_simple_query(self, query: str) -> bool:
        """간단한 쿼리 판별 로직"""
        # 짧은 키워드성 쿼리
        if len(query.strip()) < 10:
            return True
        
        # 단순 명사구
        simple_patterns = [
            lambda q: len(q.split()) <= 2,  # 2단어 이하
            lambda q: not any(char in q for char in '?!'),  # 질문/감탄문 아님
            lambda q: not any(word in q for word in ['어떻게', '왜', '무엇', '언제', '어디서'])
        ]
        
        return sum(pattern(query) for pattern in simple_patterns) >= 2
    
    async def _call_gpt5_nano(self, query: str) -> Optional[Dict[str, Any]]:
        """GPT-5-nano API 호출"""
        try:
            self.stats['gpt5_api_calls'] += 1
            
            # GPT-5-nano 최적화된 파라미터
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-5-nano",  # GPT-5 nano 모델
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Korean query expansion specialist. Always respond in valid JSON format."
                    },
                    {
                        "role": "user", 
                        "content": self.expansion_prompt.format(query=query)
                    }
                ],
                # GPT-5 특화 파라미터
                temperature=0.3,  # 일관된 결과를 위해 낮은 temperature
                max_tokens=1000,  # 구조화된 응답에 충분한 토큰
                response_format={"type": "json_object"},  # JSON 응답 강제
                # GPT-5의 새로운 매개변수들
                # reasoning={"effort": "low"},  # 빠른 추론
                # text={"verbosity": "medium"}  # 적절한 상세도
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"GPT-5-nano 응답 길이: {len(content)}자")
            
            # JSON 파싱
            try:
                parsed_json = json.loads(content)
                return parsed_json
            except json.JSONDecodeError as je:
                logger.error(f"JSON 파싱 실패: {je}")
                self.stats['json_parse_failures'] += 1
                return None
                
        except Exception as e:
            logger.error(f"GPT-5-nano API 호출 오류: {e}")
            return None
    
    def _parse_expansion_result(
        self, 
        raw_data: Dict[str, Any], 
        original_query: str
    ) -> ExpandedQuery:
        """GPT-5 응답을 구조화된 객체로 변환"""
        try:
            return ExpandedQuery(
                original_query=original_query,
                synonyms=raw_data.get('synonyms', []),
                related_terms=raw_data.get('related_terms', []),
                core_keywords=raw_data.get('core_keywords', []),
                intent=SearchIntent(raw_data.get('intent', 'factual')),
                complexity=QueryComplexity(raw_data.get('complexity', 'medium')),
                expanded_queries=raw_data.get('expanded_queries', []),
                search_strategy=raw_data.get('search_strategy', 'hybrid')
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"확장 결과 파싱 오류: {e}, 폴백 사용")
            return self._create_fallback_expansion(original_query)
    
    def _create_simple_expansion(self, query: str) -> ExpandedQuery:
        """간단한 쿼리용 빠른 확장"""
        keywords = query.split()
        
        return ExpandedQuery(
            original_query=query,
            synonyms=[],
            related_terms=[],
            core_keywords=[
                {"keyword": kw, "weight": 1.0 - (i * 0.1)} 
                for i, kw in enumerate(keywords[:3])
            ],
            intent=SearchIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            expanded_queries=[
                {"query": query, "weight": 1.0, "focus": "original"}
            ],
            search_strategy="focused"
        )
    
    def _create_fallback_expansion(self, query: str) -> ExpandedQuery:
        """폴백 확장 (GPT-5 실패시)"""
        keywords = query.split()
        
        return ExpandedQuery(
            original_query=query,
            synonyms=[],
            related_terms=[],
            core_keywords=[
                {"keyword": kw, "weight": 0.8} for kw in keywords
            ],
            intent=SearchIntent.FACTUAL,
            complexity=QueryComplexity.MEDIUM,
            expanded_queries=[
                {"query": query, "weight": 1.0, "focus": "original"},
                {"query": " ".join(keywords), "weight": 0.8, "focus": "keywords"}
            ],
            search_strategy="hybrid"
        )
    
    def _update_stats(self, expanded_query: ExpandedQuery, processing_time: float):
        """통계 업데이트"""
        self.stats['successful_expansions'] += 1
        self.stats['complexity_distribution'][expanded_query.complexity.value] += 1
        self.stats['intent_distribution'][expanded_query.intent.value] += 1
        
        # 평균 응답 시간 업데이트
        total_expansions = self.stats['successful_expansions']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total_expansions - 1) + processing_time) / total_expansions
        )
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """확장 성능 통계"""
        success_rate = (
            self.stats['successful_expansions'] / max(1, self.stats['total_expansions'])
        ) * 100
        
        return {
            **self.stats,
            'success_rate_percentage': round(success_rate, 2),
            'average_response_time_ms': round(self.stats['average_response_time'] * 1000, 2)
        }

# 확장 쿼리 적용을 위한 검색 인터페이스
class EnhancedSearchEngine:
    """확장된 쿼리를 활용하는 검색 엔진"""
    
    def __init__(self, query_expander: GPT5QueryExpansionEngine, retrieval_module):
        self.query_expander = query_expander
        self.retrieval_module = retrieval_module
    
    async def search_with_expansion(
        self, 
        original_query: str, 
        options: Dict[str, Any] = None
    ) -> List[Any]:
        """확장된 쿼리를 사용한 검색"""
        options = options or {}
        
        # 1. 쿼리 확장
        logger.info(f"쿼리 확장 시작: {original_query}")
        expanded_query = await self.query_expander.expand_query(original_query)
        
        if not expanded_query:
            logger.warning("쿼리 확장 실패, 원본 쿼리로 검색")
            return await self.retrieval_module.search(original_query, options)
        
        # 2. 다중 쿼리 검색 전략
        all_results = []
        search_strategy = expanded_query.search_strategy
        
        if search_strategy == "focused":
            # 원본 쿼리만 사용
            results = await self.retrieval_module.search(original_query, options)
            all_results.extend(results)
            
        elif search_strategy == "broad":
            # 모든 확장 쿼리 사용
            for exp_query in expanded_query.expanded_queries:
                query_text = exp_query["query"]
                weight = exp_query["weight"]
                
                # 가중치에 따른 결과 수 조정
                adjusted_limit = int(options.get('limit', 25) * weight)
                query_options = {**options, 'limit': max(adjusted_limit, 5)}
                
                results = await self.retrieval_module.search(query_text, query_options)
                
                # 결과에 확장 메타데이터 추가
                for result in results:
                    if hasattr(result, 'metadata'):
                        result.metadata['expansion_weight'] = weight
                        result.metadata['expansion_query'] = query_text
                
                all_results.extend(results)
        
        elif search_strategy == "hybrid":
            # 원본 + 상위 2개 확장 쿼리
            queries_to_search = [original_query]
            queries_to_search.extend([
                eq["query"] for eq in 
                sorted(expanded_query.expanded_queries, key=lambda x: x["weight"], reverse=True)[:2]
            ])
            
            for query in queries_to_search:
                results = await self.retrieval_module.search(query, options)
                all_results.extend(results)
        
        # 3. 중복 제거 및 스코어 조정
        unique_results = self._deduplicate_results(all_results)
        
        logger.info(
            f"확장 검색 완료: {len(all_results)}개 원본 → {len(unique_results)}개 고유 결과"
        )
        
        return unique_results[:options.get('limit', 25)]
    
    def _deduplicate_results(self, results: List[Any]) -> List[Any]:
        """결과 중복 제거"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(getattr(result, 'content', str(result))[:200])
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results

# 사용 예시 및 테스트
async def example_usage():
    """사용 예시"""
    config = {
        'llm': {
            'openai': {
                'api_key': 'your-openai-api-key'
            }
        }
    }
    
    # 쿼리 확장 엔진 초기화
    query_expander = GPT5QueryExpansionEngine(config)
    
    # 테스트 쿼리들
    test_queries = [
        "머신러닝 알고리즘",                    # 간단
        "딥러닝과 머신러닝의 차이점은 무엇인가?",     # 비교
        "파이썬으로 웹 크롤링하는 방법",            # 절차적
        "인공지능 윤리 문제",                    # 복합적
    ]
    
    for query in test_queries:
        print(f"\n=== 쿼리: {query} ===")
        
        expanded = await query_expander.expand_query(query)
        if expanded:
            print(f"의도: {expanded.intent.value}")
            print(f"복잡도: {expanded.complexity.value}")
            print(f"핵심 키워드: {expanded.core_keywords}")
            print(f"확장 쿼리 수: {len(expanded.expanded_queries)}")
        
        print("-" * 50)

# 설정 업데이트 (config.yaml)
config_yaml_update = """
# GPT-5 쿼리 확장 설정
query_expansion:
  enabled: true
  provider: "gpt-5-nano"
  fallback_enabled: true
  simple_query_threshold: 10  # 10자 이하는 간단한 쿼리로 처리
  max_expanded_queries: 5     # 최대 확장 쿼리 수
  
# 기존 LLM 설정에 추가
llm:
  openai:
    api_key: "${OPENAI_API_KEY}"
    models:
      expansion: "gpt-5-nano"  # 쿼리 확장 전용
      generation: "gpt-5"      # 답변 생성 전용
"""

if __name__ == "__main__":
    asyncio.run(example_usage())
# 📐 Gemini Embedding 001 마이그레이션 설계서

## 🎯 목표
Google text-embedding-004 (768차원)에서 gemini-embedding-001 (1536차원)로 전환하여 더 높은 차원의 의미적 표현력 확보

## 🔄 마이그레이션 아키텍처

### 1. 현재 시스템 구조
```yaml
현재 임베딩:
  모델: text-embedding-004
  차원: 768
  제공자: Google (langchain_google_genai)
  정규화: 자동 (API 레벨)
  
Qdrant 설정:
  dense_vector_size: 768
  distance: cosine
  하이브리드: dense(60%) + sparse(40%)
```

### 2. 목표 시스템 구조
```yaml
새로운 임베딩:
  모델: gemini-embedding-001
  차원: 1536
  제공자: Google Generative AI
  정규화: 수동 L2 정규화 필요
  task_type: 
    - RETRIEVAL_DOCUMENT (문서)
    - RETRIEVAL_QUERY (쿼리)
    
Qdrant 설정:
  dense_vector_size: 1536 (2배 증가)
  distance: cosine (유지)
  하이브리드: dense(60%) + sparse(40%) (유지)
```

## 📝 구현 계획

### Phase 1: 임베딩 클래스 구현
```python
# app/modules/gemini_embeddings.py

import google.generativeai as genai
import numpy as np
from typing import List, Literal
from langchain.embeddings.base import Embeddings

class GeminiEmbeddings(Embeddings):
    """Gemini Embedding 001 모델 래퍼"""
    
    def __init__(
        self,
        google_api_key: str,
        model_name: str = "models/gemini-embedding-001",
        output_dimensionality: int = 1536,
        task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "RETRIEVAL_DOCUMENT"
    ):
        genai.configure(api_key=google_api_key)
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """L2 정규화 수행"""
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return (arr / norm).tolist()
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 (RETRIEVAL_DOCUMENT)"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality
            )
            normalized = self._normalize_vector(result['embedding'])
            embeddings.append(normalized)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 (RETRIEVAL_QUERY)"""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.output_dimensionality
        )
        return self._normalize_vector(result['embedding'])
```

### Phase 2: 설정 파일 업데이트
```yaml
# config.yaml 변경사항
embeddings:
  provider: "gemini"  # google -> gemini
  model: "models/gemini-embedding-001"  # 변경
  output_dimensionality: 1536  # 추가
  batch_size: 100
  
qdrant:
  vector_size: 1536  # 768 -> 1536
  # 나머지 설정 유지
```

### Phase 3: DocumentProcessor 수정
```python
# document_processing.py 수정
def init_embedders(self):
    if provider == 'gemini':
        from .gemini_embeddings import GeminiEmbeddings
        self.embedder = GeminiEmbeddings(
            google_api_key=api_key,
            output_dimensionality=1536
        )
```

### Phase 4: Qdrant 컬렉션 재생성
```python
# retrieval_rerank.py 수정
async def _init_collection(self):
    # Dense 벡터 크기 1536으로 변경
    test_embedding = await asyncio.to_thread(
        self.embedder.embed_query, "test"
    )
    dense_vector_size = len(test_embedding)  # 1536
    
    vectors_config = VectorParams(
        size=dense_vector_size,  # 1536
        distance=Distance.COSINE
    )
```

## ⚠️ 주요 고려사항

### 1. L2 정규화 필수
- gemini-embedding-001의 1536차원 출력은 정규화되지 않음
- 코사인 유사도 계산을 위해 반드시 L2 정규화 수행

### 2. Task Type 구분
- 문서: `RETRIEVAL_DOCUMENT`
- 쿼리: `RETRIEVAL_QUERY`
- 동일한 task_type 간 비교가 가장 정확

### 3. 토큰 제한
- 최대 2,048 토큰 입력 제한
- 긴 문서는 청킹 필요

### 4. 성능 영향
- 벡터 크기 2배 증가 → 저장 공간 2배
- 검색 속도 약간 감소 예상
- 의미적 표현력 향상으로 정확도 개선 기대

## 🔄 마이그레이션 단계

### Step 1: 백업
```bash
# 기존 Qdrant 데이터 백업
qdrant-client export --collection documents --path ./backup
```

### Step 2: 코드 배포
1. GeminiEmbeddings 클래스 구현
2. config.yaml 업데이트
3. 모듈 수정 및 테스트

### Step 3: 데이터 재색인
```python
# 기존 문서 재임베딩
async def reindex_all_documents():
    # 1. 기존 문서 추출
    # 2. 새 임베딩 생성 (1536차원)
    # 3. Qdrant 재업로드
```

### Step 4: 검증
- 임베딩 차원 확인 (1536)
- L2 norm = 1.0 확인
- 검색 정확도 비교 테스트

## 📊 예상 효과

### 장점
- **2배 높은 차원**: 더 풍부한 의미적 표현
- **최신 모델**: gemini-embedding-001의 개선된 성능
- **Task 최적화**: 문서/쿼리별 최적화된 임베딩

### 단점
- **저장 공간**: 2배 증가 (768 → 1536)
- **처리 시간**: 약간의 성능 저하
- **재색인 필요**: 기존 데이터 전체 재처리

## 🚀 구현 우선순위

1. **[필수] GeminiEmbeddings 클래스 구현**
2. **[필수] L2 정규화 로직 추가**
3. **[필수] config.yaml 및 모듈 수정**
4. **[필수] Qdrant 컬렉션 재생성**
5. **[선택] 배치 처리 최적화**
6. **[선택] 성능 모니터링 추가**

## 🧪 테스트 계획

### 단위 테스트
```python
def test_embedding_dimension():
    embeddings = GeminiEmbeddings(...)
    vector = embeddings.embed_query("test")
    assert len(vector) == 1536
    assert abs(np.linalg.norm(vector) - 1.0) < 0.001
```

### 통합 테스트
- Qdrant 업로드/검색 테스트
- 하이브리드 검색 동작 확인
- 리랭킹 호환성 검증

### 성능 테스트
- 임베딩 생성 시간 측정
- 검색 속도 비교
- 메모리 사용량 모니터링
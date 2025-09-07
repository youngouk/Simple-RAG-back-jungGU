"""
Gemini Embedding API wrapper for 1536-dimensional embeddings
gemini-embedding-001 모델을 사용한 1536차원 임베딩 구현
"""
import os
import asyncio
from typing import List, Literal, Optional
import numpy as np
import google.generativeai as genai
from langchain.embeddings.base import Embeddings

from ..lib.logger import get_logger

logger = get_logger(__name__)

class GeminiEmbeddings(Embeddings):
    """
    Google Gemini Embedding 001 모델 래퍼
    1536차원 임베딩 벡터 생성 및 L2 정규화 수행
    """
    
    def __init__(
        self,
        google_api_key: str,
        model_name: str = "models/gemini-embedding-001",
        output_dimensionality: int = 1536,
        batch_size: int = 100,
        task_type: Optional[Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]] = None
    ):
        """
        Initialize Gemini Embeddings
        
        Args:
            google_api_key: Google API key
            model_name: Model name (default: models/gemini-embedding-001)
            output_dimensionality: Output dimension (default: 1536)
            batch_size: Batch size for embedding generation
            task_type: Default task type (can be overridden in methods)
        """
        # API 키 설정
        genai.configure(api_key=google_api_key)
        
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size
        self.default_task_type = task_type or "RETRIEVAL_DOCUMENT"
        
        logger.info(f"Initialized GeminiEmbeddings with model: {model_name}, dimensions: {output_dimensionality}")
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        L2 정규화 수행 (필수)
        1536차원 출력은 정규화되지 않은 상태로 반환되므로 반드시 정규화 필요
        
        Args:
            vector: 정규화할 벡터
            
        Returns:
            L2 정규화된 벡터
        """
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        
        if norm > 0:
            normalized = arr / norm
            return normalized.tolist()
        
        logger.warning("Zero norm vector encountered, returning as-is")
        return vector
    
    def _batch_embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        배치 단위로 임베딩 생성 (동기 버전)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            task_type: Task type for embedding
            
        Returns:
            정규화된 임베딩 벡터 리스트
        """
        embeddings = []
        
        # 배치 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Gemini API 호출
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality
                )
                
                # 결과가 단일 임베딩인 경우와 리스트인 경우 처리
                if 'embedding' in result:
                    # 배치가 1개인 경우와 여러 개인 경우 구분
                    if len(batch) == 1:
                        # 단일 텍스트 처리
                        normalized = self._normalize_vector(result['embedding'])
                        embeddings.append(normalized)
                    else:
                        # 여러 텍스트를 한 번에 처리한 경우
                        # result['embedding']은 리스트의 리스트
                        for embedding in result['embedding']:
                            normalized = self._normalize_vector(embedding)
                            embeddings.append(normalized)
                elif 'embeddings' in result:
                    # 이 경우는 실제로 발생하지 않지만 안전을 위해 유지
                    for embedding in result['embeddings']:
                        normalized = self._normalize_vector(embedding)
                        embeddings.append(normalized)
                else:
                    logger.error(f"Unexpected result format: {result.keys()}")
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//self.batch_size}: {e}")
                # 오류 발생 시 빈 벡터 추가
                for _ in batch:
                    embeddings.append([0.0] * self.output_dimensionality)
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 임베딩 생성 (RETRIEVAL_DOCUMENT 타입)
        
        Args:
            texts: 임베딩할 문서 텍스트 리스트
            
        Returns:
            L2 정규화된 1536차원 임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents with task_type=RETRIEVAL_DOCUMENT")
        
        # 배치 처리로 임베딩 생성
        embeddings = self._batch_embed(texts, "RETRIEVAL_DOCUMENT")
        
        logger.info(f"Generated {len(embeddings)} document embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 임베딩 생성 (RETRIEVAL_QUERY 타입)
        
        Args:
            text: 임베딩할 쿼리 텍스트
            
        Returns:
            L2 정규화된 1536차원 임베딩 벡터
        """
        logger.debug(f"Embedding query with task_type=RETRIEVAL_QUERY")
        
        try:
            # 단일 쿼리 임베딩
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality
            )
            
            # L2 정규화 수행
            embedding = result.get('embedding', [])
            normalized = self._normalize_vector(embedding)
            
            # 차원 확인
            if len(normalized) != self.output_dimensionality:
                logger.warning(f"Unexpected embedding dimension: {len(normalized)} != {self.output_dimensionality}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # 오류 발생 시 영벡터 반환
            return [0.0] * self.output_dimensionality
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        비동기 문서 임베딩 생성
        
        Args:
            texts: 임베딩할 문서 텍스트 리스트
            
        Returns:
            L2 정규화된 1536차원 임베딩 벡터 리스트
        """
        # 동기 메서드를 비동기로 실행
        return await asyncio.to_thread(self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        비동기 쿼리 임베딩 생성
        
        Args:
            text: 임베딩할 쿼리 텍스트
            
        Returns:
            L2 정규화된 1536차원 임베딩 벡터
        """
        # 동기 메서드를 비동기로 실행
        return await asyncio.to_thread(self.embed_query, text)
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        임베딩 벡터 검증
        
        Args:
            embedding: 검증할 임베딩 벡터
            
        Returns:
            검증 성공 여부
        """
        # 차원 확인
        if len(embedding) != self.output_dimensionality:
            logger.error(f"Invalid dimension: {len(embedding)} != {self.output_dimensionality}")
            return False
        
        # L2 norm 확인 (정규화 여부)
        norm = np.linalg.norm(np.array(embedding))
        if abs(norm - 1.0) > 0.01:  # 허용 오차
            logger.warning(f"Vector not normalized: norm={norm}")
            return False
        
        return True
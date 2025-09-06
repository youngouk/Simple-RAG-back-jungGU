# 시멘틱 청킹 전략 설계
# Semantic Chunking Strategy Design

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """청킹 설정"""
    target_chunk_size: int = 1250        # 목표 청크 크기 (1,000-1,500자 중간값)
    min_chunk_size: int = 1000           # 최소 청크 크기
    max_chunk_size: int = 1500           # 최대 청크 크기
    semantic_threshold: float = 0.3      # 의미적 분할 임계값
    overlap_sentences: int = 2           # 오버랩할 문장 수
    sentence_window: int = 3             # 의미 분석 윈도우 크기
    preserve_structure: bool = True      # 문서 구조 보존 여부
    korean_optimized: bool = True        # 한국어 최적화
    
class BaseSplitter(ABC):
    """기본 분할기 인터페이스"""
    
    @abstractmethod
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서 분할"""
        pass

class SemanticSplitter(BaseSplitter):
    """시멘틱 기반 문서 분할기"""
    
    def __init__(self, config: ChunkingConfig, embedder: GoogleGenerativeAIEmbeddings):
        self.config = config
        self.embedder = embedder
        self.sentence_splitter = self._init_sentence_splitter()
        
        # 통계 추적
        self.stats = {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'semantic_splits': 0,
            'structure_preserving_splits': 0,
            'fallback_splits': 0
        }
    
    def _init_sentence_splitter(self) -> re.Pattern:
        """한국어 최적화 문장 분리기 초기화"""
        if self.config.korean_optimized:
            # 한국어 문장 종료 패턴
            pattern = r'[.!?](?=\s|$)|(?<=[다가나])\.(?=\s|$)|(?<=[요다])\.(?=\s|$)|(?<=다)\s*\n|(?<=요)\s*\n'
        else:
            # 기본 문장 분리 패턴
            pattern = r'[.!?]+\s*'
        
        return re.compile(pattern)
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 시멘틱 청크로 분할"""
        if not documents:
            return []
        
        logger.info(f"시멘틱 청킹 시작: {len(documents)}개 문서")
        
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            try:
                # 문서별 청킹 처리
                doc_chunks = await self._split_single_document(document, doc_idx)
                all_chunks.extend(doc_chunks)
                
            except Exception as e:
                logger.error(f"문서 {doc_idx} 청킹 실패: {e}")
                # 폴백: RecursiveCharacterTextSplitter 사용
                fallback_chunks = await self._fallback_split(document, doc_idx)
                all_chunks.extend(fallback_chunks)
                self.stats['fallback_splits'] += len(fallback_chunks)
        
        # 최종 통계 업데이트
        self._update_final_stats(all_chunks)
        
        logger.info(f"시멘틱 청킹 완료: {len(all_chunks)}개 청크 생성")
        logger.info(f"평균 청크 크기: {self.stats['avg_chunk_size']:.1f}자")
        
        return all_chunks
    
    async def _split_single_document(self, document: Document, doc_idx: int) -> List[Document]:
        """단일 문서 분할"""
        content = document.page_content.strip()
        
        if len(content) <= self.config.max_chunk_size:
            # 이미 적정 크기면 그대로 반환
            chunk = Document(
                page_content=content,
                metadata={
                    **document.metadata,
                    'chunk_method': 'no_split',
                    'chunk_size': len(content),
                    'doc_index': doc_idx,
                    'chunk_index': 0
                }
            )
            return [chunk]
        
        # 1단계: 구조 기반 분할 (제목, 단락 등)
        if self.config.preserve_structure:
            structural_chunks = await self._split_by_structure(content)
            if structural_chunks:
                final_chunks = []
                for i, chunk_text in enumerate(structural_chunks):
                    if len(chunk_text) > self.config.max_chunk_size:
                        # 구조 청크가 너무 크면 시멘틱 분할 적용
                        sub_chunks = await self._semantic_split(chunk_text, f"{doc_idx}.{i}")
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk_text)
                
                return self._create_chunk_documents(final_chunks, document, doc_idx, 'structural_semantic')
        
        # 2단계: 순수 시멘틱 분할
        semantic_chunks = await self._semantic_split(content, str(doc_idx))
        return self._create_chunk_documents(semantic_chunks, document, doc_idx, 'semantic')
    
    async def _split_by_structure(self, content: str) -> List[str]:
        """문서 구조 기반 분할"""
        chunks = []
        
        # 제목 기반 분할 (마크다운 스타일)
        title_pattern = r'^(#{1,6}\s+.+)$'
        lines = content.split('\n')
        current_chunk = []
        
        for line in lines:
            line = line.strip()
            if re.match(title_pattern, line) and current_chunk:
                # 새 제목을 만나면 이전 청크 저장
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:  # 이전 청크에 병합
                chunks[-1] += '\n' + chunk_text
        
        # 단락 기반 분할 (제목이 없는 경우)
        if not chunks:
            paragraphs = re.split(r'\n\s*\n', content)
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                para_size = len(paragraph)
                
                if current_size + para_size > self.config.max_chunk_size and current_chunk:
                    # 현재 청크 저장
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_size = para_size
                else:
                    current_chunk.append(paragraph)
                    current_size += para_size
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        # 구조 보존 분할 통계 업데이트
        if chunks:
            self.stats['structure_preserving_splits'] += len(chunks)
        
        return chunks
    
    async def _semantic_split(self, content: str, doc_id: str) -> List[str]:
        """시멘틱 유사도 기반 분할"""
        try:
            # 문장 단위로 분리
            sentences = self._split_into_sentences(content)
            
            if len(sentences) <= 3:
                return [content]  # 문장이 너무 적으면 분할하지 않음
            
            # 문장별 임베딩 생성 (배치 처리로 성능 최적화)
            sentence_embeddings = await self._get_sentence_embeddings(sentences)
            
            if sentence_embeddings is None:
                logger.warning(f"임베딩 생성 실패 for doc {doc_id}, 폴백 분할 사용")
                return await self._fallback_split_text(content)
            
            # 의미적 경계 찾기
            split_points = self._find_semantic_boundaries(
                sentences, sentence_embeddings
            )
            
            # 청크 생성
            chunks = self._create_semantic_chunks(sentences, split_points)
            
            # 크기 검증 및 조정
            validated_chunks = self._validate_and_adjust_chunks(chunks)
            
            self.stats['semantic_splits'] += len(validated_chunks)
            return validated_chunks
            
        except Exception as e:
            logger.error(f"시멘틱 분할 실패 for doc {doc_id}: {e}")
            return await self._fallback_split_text(content)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        sentences = self.sentence_splitter.split(text)
        
        # 빈 문장 제거 및 정리
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # 너무 짧은 문장 제외
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    async def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """문장별 임베딩 생성"""
        try:
            # 배치 크기 제한 (메모리 효율성)
            batch_size = 10
            embeddings = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_embeddings = await asyncio.to_thread(
                    self.embedder.embed_documents, batch
                )
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def _find_semantic_boundaries(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
        """의미적 경계점 찾기"""
        split_points = [0]  # 시작점
        
        window_size = self.config.sentence_window
        threshold = self.config.semantic_threshold
        
        for i in range(window_size, len(sentences) - window_size, window_size):
            # 현재 윈도우와 다음 윈도우의 평균 임베딩 계산
            current_window = embeddings[i - window_size:i]
            next_window = embeddings[i:i + window_size]
            
            current_avg = np.mean(current_window, axis=0)
            next_avg = np.mean(next_window, axis=0)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                [current_avg], [next_avg]
            )[0][0]
            
            # 유사도가 임계값 이하면 분할점으로 선택
            if similarity < threshold:
                # 청크 크기 검증
                current_chunk_text = ' '.join(sentences[split_points[-1]:i])
                if len(current_chunk_text) >= self.config.min_chunk_size:
                    split_points.append(i)
        
        split_points.append(len(sentences))  # 끝점
        return split_points
    
    def _create_semantic_chunks(self, sentences: List[str], split_points: List[int]) -> List[str]:
        """시멘틱 분할점을 기반으로 청크 생성"""
        chunks = []
        
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            # 오버랩 처리
            if i > 0 and self.config.overlap_sentences > 0:
                overlap_start = max(0, start_idx - self.config.overlap_sentences)
                chunk_sentences = sentences[overlap_start:end_idx]
            else:
                chunk_sentences = sentences[start_idx:end_idx]
            
            chunk_text = ' '.join(chunk_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def _validate_and_adjust_chunks(self, chunks: List[str]) -> List[str]:
        """청크 크기 검증 및 조정"""
        adjusted_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = len(current_chunk)
            
            # 너무 작은 청크는 다음 청크와 병합
            if current_size < self.config.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                merged_chunk = current_chunk + ' ' + next_chunk
                
                if len(merged_chunk) <= self.config.max_chunk_size:
                    adjusted_chunks.append(merged_chunk)
                    i += 2  # 두 청크를 모두 건너뛰기
                    continue
            
            # 너무 큰 청크는 RecursiveCharacterTextSplitter로 재분할
            if current_size > self.config.max_chunk_size:
                sub_chunks = self._force_split_large_chunk(current_chunk)
                adjusted_chunks.extend(sub_chunks)
            else:
                adjusted_chunks.append(current_chunk)
            
            i += 1
        
        return adjusted_chunks
    
    def _force_split_large_chunk(self, chunk: str) -> List[str]:
        """큰 청크를 강제로 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Document 객체로 감싼 후 분할
        doc = Document(page_content=chunk, metadata={})
        split_docs = splitter.split_documents([doc])
        
        return [doc.page_content for doc in split_docs]
    
    def _create_chunk_documents(self, chunks: List[str], original_doc: Document, 
                               doc_idx: int, method: str) -> List[Document]:
        """청크 텍스트를 Document 객체로 변환"""
        chunk_documents = []
        
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **original_doc.metadata,
                    'chunk_method': method,
                    'chunk_size': len(chunk_text),
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
            )
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    async def _fallback_split(self, document: Document, doc_idx: int) -> List[Document]:
        """폴백 분할 (RecursiveCharacterTextSplitter)"""
        logger.info(f"문서 {doc_idx}에 폴백 분할 적용")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=self.config.overlap_sentences * 50,  # 대략적인 문장 오버랩
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        split_docs = splitter.split_documents([document])
        
        # 메타데이터 업데이트
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_method': 'fallback_recursive',
                'chunk_size': len(doc.page_content),
                'doc_index': doc_idx,
                'chunk_index': i,
                'total_chunks': len(split_docs)
            })
        
        return split_docs
    
    async def _fallback_split_text(self, text: str) -> List[str]:
        """텍스트 폴백 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        doc = Document(page_content=text, metadata={})
        split_docs = splitter.split_documents([doc])
        
        return [doc.page_content for doc in split_docs]
    
    def _update_final_stats(self, chunks: List[Document]):
        """최종 통계 업데이트"""
        if not chunks:
            return
        
        self.stats['total_chunks'] = len(chunks)
        
        # 평균 청크 크기 계산
        total_size = sum(len(chunk.page_content) for chunk in chunks)
        self.stats['avg_chunk_size'] = total_size / len(chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """청킹 통계 반환"""
        return {
            **self.stats,
            'config': {
                'target_chunk_size': self.config.target_chunk_size,
                'min_chunk_size': self.config.min_chunk_size,
                'max_chunk_size': self.config.max_chunk_size,
                'semantic_threshold': self.config.semantic_threshold,
                'korean_optimized': self.config.korean_optimized
            }
        }

class HybridSplitter(BaseSplitter):
    """하이브리드 분할기 (시멘틱 + RecursiveCharacterTextSplitter)"""
    
    def __init__(self, config: ChunkingConfig, embedder: GoogleGenerativeAIEmbeddings):
        self.semantic_splitter = SemanticSplitter(config, embedder)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.target_chunk_size,
            chunk_overlap=config.overlap_sentences * 50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.config = config
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """하이브리드 분할: 문서 특성에 따라 적절한 분할 방법 선택"""
        if not documents:
            return []
        
        all_chunks = []
        
        for doc in documents:
            content_length = len(doc.page_content)
            
            # 문서 크기와 특성에 따라 분할 방법 선택
            if content_length > 5000:  # 긴 문서는 시멘틱 분할
                try:
                    chunks = await self.semantic_splitter._split_single_document(doc, 0)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"시멘틱 분할 실패, RecursiveCharacterTextSplitter 사용: {e}")
                    chunks = self.recursive_splitter.split_documents([doc])
                    all_chunks.extend(chunks)
            else:  # 짧은 문서는 RecursiveCharacterTextSplitter
                chunks = self.recursive_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata['chunk_method'] = 'recursive'
                all_chunks.extend(chunks)
        
        return all_chunks

# 팩토리 함수
def create_splitter(splitter_type: str, config: ChunkingConfig, 
                   embedder: GoogleGenerativeAIEmbeddings) -> BaseSplitter:
    """분할기 팩토리 함수"""
    splitter_map = {
        'semantic': SemanticSplitter,
        'hybrid': HybridSplitter,
        'recursive': lambda cfg, emb: RecursiveCharacterTextSplitter(
            chunk_size=cfg.target_chunk_size,
            chunk_overlap=cfg.overlap_sentences * 50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    }
    
    if splitter_type not in splitter_map:
        raise ValueError(f"지원하지 않는 분할기 타입: {splitter_type}")
    
    return splitter_map[splitter_type](config, embedder)

# 설정 예시
def get_default_chunking_config() -> ChunkingConfig:
    """기본 청킹 설정 반환"""
    return ChunkingConfig(
        target_chunk_size=1250,      # 1,000-1,500자 중간값
        min_chunk_size=1000,         # 최소 1,000자
        max_chunk_size=1500,         # 최대 1,500자
        semantic_threshold=0.3,      # 의미적 분할 임계값 30%
        overlap_sentences=2,         # 2문장 오버랩
        sentence_window=3,           # 3문장 윈도우로 의미 분석
        preserve_structure=True,     # 문서 구조 보존
        korean_optimized=True        # 한국어 최적화 활성화
    )

# 통합 예시 
async def example_usage():
    """사용 예시"""
    # 설정 초기화
    config = get_default_chunking_config()
    
    # 임베더 초기화 (실제 사용 시 API 키 필요)
    # embedder = GoogleGenerativeAIEmbeddings(
    #     model="text-embedding-004",
    #     google_api_key="your-api-key"
    # )
    
    # 시멘틱 분할기 생성
    # splitter = create_splitter('semantic', config, embedder)
    
    # 문서 생성 (예시)
    # documents = [Document(page_content="긴 문서 내용...", metadata={})]
    
    # 분할 실행
    # chunks = await splitter.split_documents(documents)
    
    # 통계 출력
    # if hasattr(splitter, 'get_stats'):
    #     print(f"청킹 통계: {splitter.get_stats()}")
    
    pass

if __name__ == "__main__":
    asyncio.run(example_usage())

# 로깅 설정 예시
"""
2025-01-06 15:30:15 - INFO - 시멘틱 청킹 시작: 5개 문서
2025-01-06 15:30:16 - INFO - 문서 구조 기반 분할: 3개 구조 청크 생성
2025-01-06 15:30:17 - INFO - 시멘틱 분석: 임계값 0.3 적용, 윈도우 크기 3
2025-01-06 15:30:18 - INFO - 의미적 경계 발견: [0, 15, 32, 48] 분할점
2025-01-06 15:30:19 - INFO - 청크 크기 검증: 2개 청크 병합, 1개 청크 재분할
2025-01-06 15:30:20 - INFO - 시멘틱 청킹 완료: 23개 청크 생성
2025-01-06 15:30:20 - INFO - 평균 청크 크기: 1,247.3자
2025-01-06 15:30:21 - INFO - 청킹 통계: semantic_splits=18, structure_preserving_splits=5, fallback_splits=0
"""
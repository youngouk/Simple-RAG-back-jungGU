"""
Chat API endpoints
채팅 관련 API 엔드포인트
"""
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from ..lib.logger import get_logger, create_chat_logging_middleware

logger = get_logger(__name__)
chat_logger = create_chat_logging_middleware()
router = APIRouter()

# Dependencies (will be injected from main.py)
modules: Dict[str, Any] = {}
config: Dict[str, Any] = {}

def set_dependencies(app_modules: Dict[str, Any], app_config: Dict[str, Any]):
    """의존성 주입"""
    global modules, config
    modules = app_modules
    config = app_config

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., min_length=1, max_length=1000, description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")
    stream: bool = Field(False, description="스트리밍 응답 여부")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 옵션")

    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class Source(BaseModel):
    """소스 정보 모델"""
    id: int
    document: str
    page: Optional[int] = None
    chunk: Optional[int] = None
    relevance: float
    content_preview: str

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str
    sources: List[Source]
    session_id: str
    processing_time: float
    tokens_used: int
    timestamp: str
    model_info: Optional[Dict[str, Any]] = None

class SessionCreateRequest(BaseModel):
    """세션 생성 요청 모델"""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SessionResponse(BaseModel):
    """세션 응답 모델"""
    session_id: str
    message: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    """채팅 히스토리 응답 모델"""
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    limit: int
    offset: int
    has_more: bool

class StatsResponse(BaseModel):
    """통계 응답 모델"""
    chat: Dict[str, Any]
    session: Dict[str, Any]
    timestamp: str

# 통계 정보
stats = {
    "total_chats": 0,
    "total_tokens": 0,
    "average_latency": 0.0,
    "error_rate": 0.0,
    "errors": 0
}

def update_stats(data: Dict[str, Any]):
    """통계 업데이트"""
    stats["total_chats"] += 1
    
    if data.get("success"):
        if data.get("tokens_used"):
            stats["total_tokens"] += data["tokens_used"]
            
        if data.get("latency"):
            current_avg = stats["average_latency"]
            chat_count = stats["total_chats"]
            stats["average_latency"] = (current_avg * (chat_count - 1) + data["latency"]) / chat_count
    else:
        stats["errors"] += 1
        stats["error_rate"] = (stats["errors"] / stats["total_chats"]) * 100

def get_request_context(request: Request) -> Dict[str, Any]:
    """요청 컨텍스트 추출"""
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "referrer": request.headers.get("referer")
    }

async def handle_session(session_id: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
    """세션 처리 - 개선된 버전 (세션 ID 불일치 방지 강화)"""
    try:
        session_module = modules.get('session')
        if not session_module:
            raise HTTPException(status_code=500, detail="Session module not available")
        
        # 요청된 session_id 로깅 (디버깅용)
        logger.info(f"🔍 세션 요청 - 요청받은 session_id: {session_id}")
            
        if session_id:
            # 기존 세션 조회
            logger.info(f"기존 세션 조회 시도: {session_id}")
            logger.debug(f"세션 모듈의 현재 세션들: {list(session_module.sessions.keys())}")
            session_result = await session_module.get_session(session_id, context)
            
            if session_result.get("is_valid"):
                # 중요: 원래 요청된 session_id를 반드시 유지!
                logger.info(f"✅ 세션 유효함 - 요청 ID: {session_id}, 응답 ID: {session_id}")
                
                # 방어적 검증: 요청과 응답 ID가 일치하는지 확인
                result_session_id = session_result.get('renewed_session_id', session_id)
                if result_session_id != session_id:
                    logger.error(f"🚨 세션 ID 불일치 감지! 요청: {session_id}, 응답: {result_session_id}")
                
                return {
                    "success": True,
                    "session_id": session_id,  # 무조건 원본 session_id 사용
                    "is_new": False,
                    "validation_result": session_result
                }
            else:
                logger.warning(f"세션 만료/없음: {session_id}, 이유: {session_result.get('reason', 'unknown')}")
                # 세션이 만료되거나 없을 때 사용자에게 알림
                logger.info(f"새 세션 생성 중... (기존 세션: {session_id})")
        
        # 새 세션 생성 - 사용자 제공 세션 ID 사용
        logger.debug(f"새 세션 생성 전 - 세션 모듈 ID: {id(session_module)}")
        new_session = await session_module.create_session({"metadata": context}, session_id=session_id)
        new_session_id = new_session["session_id"]
        
        logger.info(f"✅ 새 세션 생성 완료 - 생성된 session_id: {new_session_id}")
        logger.debug(f"새 세션 생성 후 - 전체 세션 수: {len(session_module.sessions)}")
        logger.debug(f"새 세션 생성 후 - 세션 키 목록: {list(session_module.sessions.keys())}")
        
        return {
            "success": True,
            "session_id": new_session_id,
            "is_new": True,
            "message": "새 대화 세션이 시작되었습니다."
        }
        
    except Exception as e:
        logger.error(f"Session handling error: {e}")
        return {
            "success": False,
            "message": "Failed to handle session"
        }

def extract_topic(message: str) -> str:
    """토픽 추출 (간단한 키워드 기반)"""
    # 안전한 메시지 처리
    if isinstance(message, list):
        message = ' '.join(str(item) for item in message)
    elif not isinstance(message, str):
        message = str(message)
    
    if not message:
        return 'general'
    
    keywords = {
        'search': ['검색', '찾기', '찾아', '검색해'],
        'document': ['문서', '파일', '자료', '데이터'],
        'help': ['도움', '도와', '설명', '알려'],
        'technical': ['기술', '개발', '코드', '프로그래밍'],
        'general': ['일반', '기본', '소개', '개요']
    }
    
    try:
        lower_message = message.lower()
        
        for topic, words in keywords.items():
            if any(word in lower_message for word in words):
                return topic
        
        return 'general'
    except Exception:
        # 오류 시 기본값 반환
        return 'general'

async def execute_rag_pipeline(message: str, session_id: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """RAG 파이프라인 실행"""
    start_time = time.time()
    options = options or {}
    
    logger.info("RAG Pipeline Starting", 
               message_preview=message[:50],
               session_id=session_id,
               has_retrieval_module=bool(modules.get('retrieval')),
               has_generation_module=bool(modules.get('generation')))
    
    try:
        session_module = modules.get('session')
        retrieval_module = modules.get('retrieval')
        generation_module = modules.get('generation')
        
        if not all([session_module, retrieval_module, generation_module]):
            raise Exception("Required modules not available")
        
        # 1. 대화 컨텍스트 조회 - 디버깅 추가
        logger.debug("Step 1: Getting session context...")
        
        # 세션 상태 디버깅
        logger.debug(f"세션 모듈 ID: {id(session_module)}")
        logger.debug(f"전체 세션 수: {len(session_module.sessions)}")
        logger.debug(f"세션 키 목록: {list(session_module.sessions.keys())}")
        logger.debug(f"요청된 세션 ID: {session_id}")
        
        context_string = await session_module.get_context_string(session_id)
        logger.debug("Session context retrieved", has_context=bool(context_string))
        
        # 2. 검색 실행
        logger.debug("Step 2: Starting document retrieval...")
        try:
            retrieval_config = config.get("retrieval", {})
            search_results = await retrieval_module.search(message, {
                "limit": options.get("max_sources", retrieval_config.get("max_sources", 20)),
                "min_score": options.get("min_score", retrieval_config.get("min_score", 0.01)),
                "context_string": context_string
            })
            logger.debug("Document retrieval completed", result_count=len(search_results))
        except Exception as search_error:
            search_error.step = "document_retrieval"
            raise search_error
        
        # 3. 리랭킹 실행 (설정된 경우)
        ranked_results = search_results
        if retrieval_config.get("enable_reranking", False) and search_results:
            logger.debug("Starting reranking...")
            try:
                ranked_results = await retrieval_module.rerank(message, search_results, {
                    "top_k": options.get("top_k", 10),  # 리랭킹 후 10개로 변경
                    "min_score": options.get("min_score", 0.15)  # 15% 이하 필터링
                })
                logger.debug("Reranking completed successfully",
                           original_count=len(search_results),
                           reranked_count=len(ranked_results))
            except Exception as reranker_error:
                logger.warning("Reranking failed, using original search results",
                             error=str(reranker_error))
                ranked_results = search_results
        else:
            logger.debug("Reranking disabled, using search results directly",
                        enabled=retrieval_config.get("enable_reranking", False),
                        result_count=len(search_results))
        
        # 4. 답변 생성
        logger.debug("Step 4: Starting answer generation...",
                    context_count=len(ranked_results),
                    has_session_context=bool(context_string))
        
        try:
            generation_result = await generation_module.generate_answer(message, ranked_results, {
                "session_context": context_string,
                "style": options.get("response_style", "standard"),
                "max_tokens": options.get("max_tokens", 2000)
            })
            
            # 안전한 방식으로 텍스트 존재 여부 확인
            answer_text = getattr(generation_result, 'answer', '') or getattr(generation_result, 'text', '')
            if isinstance(answer_text, list):
                answer_text = ' '.join(str(item) for item in answer_text)
            
            logger.info("Answer generation completed",
                       result_type=type(generation_result).__name__,
                       has_result=bool(generation_result),
                       has_text=bool(answer_text),
                       tokens_used=getattr(generation_result, 'tokens_used', 0))
                       
        except Exception as generation_error:
            logger.error("Answer generation error", error=str(generation_error))
            generation_error.step = "answer_generation"
            raise generation_error
        
        # 5. 결과 포맷팅
        logger.debug("Step 5: Starting result formatting...")
        sources = []
        try:
            # 최대 10개까지 표시하되, 15% 이상인 문서만 포함
            for index, doc in enumerate(ranked_results[:options.get("max_sources", 10)]):
                try:
                    # 메타데이터 구조 분석
                    if index == 0:
                        # 안전한 keys 추출
                        doc_keys = None
                        try:
                            if hasattr(doc, 'keys'):
                                # doc.keys()가 이미 리스트일 수 있으므로 안전하게 처리
                                keys_obj = doc.keys()
                                if isinstance(keys_obj, list):
                                    doc_keys = keys_obj
                                else:
                                    doc_keys = list(keys_obj)
                        except Exception as keys_error:
                            logger.debug(f"Could not extract keys from document: {keys_error}")
                            doc_keys = None
                        
                        logger.info("Document metadata structure analysis",
                                   doc_type=type(doc).__name__,
                                   doc_keys=doc_keys)
                    
                    # 다양한 메타데이터 위치에서 정보 추출
                    original_doc = getattr(doc, '_original', doc)
                    metadata = {}
                    
                    if hasattr(original_doc, 'payload') and original_doc.payload:
                        metadata = getattr(original_doc.payload, 'metadata', {})
                    elif hasattr(original_doc, 'metadata'):
                        metadata = original_doc.metadata
                    elif hasattr(doc, 'metadata'):
                        metadata = doc.metadata
                    
                    # 유사도 점수 가져오기 (0~1 범위)
                    raw_score = getattr(doc, 'score', 0)
                    
                    # 15% 미만 문서는 제외
                    if raw_score < 0.15:
                        continue
                    
                    # 콘텐츠 안전하게 추출하여 문자열로 변환
                    doc_content = getattr(doc, 'content', '') or ''
                    if isinstance(doc_content, list):
                        logger.debug(f"Document {index} content is list type, converting to string")
                        doc_content = ' '.join(str(item) for item in doc_content)
                    elif not isinstance(doc_content, str):
                        logger.debug(f"Document {index} content type: {type(doc_content)}, converting to string")
                        doc_content = str(doc_content)
                    
                    logger.debug(f"Processing document {index} - score: {raw_score}, content length: {len(doc_content)}")
                    
                    sources.append(Source(
                        id=index + 1,
                        document=metadata.get('source_file') or metadata.get('source') or 
                                metadata.get('filename') or metadata.get('document_id') or 'Unknown',
                        page=metadata.get('page_number') or metadata.get('page'),
                        chunk=metadata.get('chunk_index') or metadata.get('chunk'),
                        relevance=raw_score,  # 정규화된 점수 그대로 사용
                        content_preview=doc_content[:150] + '...' if doc_content else 'No content'
                    ))
                except Exception as doc_error:
                    logger.error(f"Error processing document {index}: {doc_error}")
                    continue
        except Exception as sources_error:
            logger.error(f"Error in sources processing: {sources_error}")
            sources = []
        
        # 안전한 방식으로 답변 추출
        logger.debug("Step 6: Extracting final answer...")
        try:
            logger.debug(f"Generation result type: {type(generation_result)}")
            
            # 안전한 방식으로 속성 확인
            try:
                attrs = dir(generation_result)
                logger.debug(f"Generation result attributes: {attrs}")
            except Exception as attr_error:
                logger.debug(f"Could not get attributes: {attr_error}")
            
            final_answer = getattr(generation_result, 'answer', '') or getattr(generation_result, 'text', '')
            logger.debug(f"Final answer type: {type(final_answer)}")
            
            if isinstance(final_answer, list):
                logger.debug("Final answer is list type, converting to string")
                final_answer = ' '.join(str(item) for item in final_answer)
            
            logger.debug(f"Final answer length: {len(final_answer) if final_answer else 0}")
            
            # 결과 딕셔너리 생성
            logger.debug("Step 7: Creating result dictionary...")
            # 안전한 길이 계산
            search_count = len(search_results) if search_results else 0
            ranked_count = len(ranked_results) if ranked_results else 0
            
            result_dict = {
                "answer": final_answer,
                "sources": sources,
                "tokens_used": getattr(generation_result, 'tokens_used', 0),
                "topic": extract_topic(message),
                "processing_time": time.time() - start_time,
                "search_results": search_count,
                "ranked_results": ranked_count,
                "model_info": {
                    "provider": getattr(generation_result, 'provider', 'unknown'),
                    "model": getattr(generation_result, 'model_used', 'unknown'),
                    "generation_time": getattr(generation_result, 'generation_time', 0),
                    "model_config": getattr(generation_result, 'model_config', {})
                }
            }
            logger.debug("Result dictionary created successfully")
            return result_dict
            
        except Exception as final_error:
            logger.error(f"Error in final answer processing: {final_error}")
            raise final_error
        
    except Exception as error:
        logger.error("RAG pipeline error",
                    error=str(error),
                    step=getattr(error, 'step', 'unknown'),
                    processing_time=time.time() - start_time)
        
        # 폴백 응답
        return {
            "answer": "죄송합니다. 현재 요청을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            "sources": [],
            "tokens_used": 0,
            "topic": extract_topic(message),
            "processing_time": time.time() - start_time,
            "search_results": 0,
            "ranked_results": 0,
            "error": True,
            "error_message": str(error)
        }

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("100/15minutes")
async def chat(request: Request, chat_request: ChatRequest):
    """채팅 처리 엔드포인트"""
    start_time = time.time()
    session_id = None
    
    try:
        # 1. 세션 처리
        context = get_request_context(request)
        session_result = await handle_session(chat_request.session_id, context)
        
        if not session_result["success"]:
            raise HTTPException(status_code=400, detail=session_result.get("message", "Session error"))
        
        session_id = session_result["session_id"]
        
        # 2. RAG 파이프라인 실행
        rag_result = await execute_rag_pipeline(chat_request.message, session_id, chat_request.options)
        
        # 3. 세션에 대화 기록
        session_module = modules.get('session')
        if session_module:
            logger.debug(f"대화 추가 전 - 세션 모듈 ID: {id(session_module)}")
            logger.debug(f"대화 추가 전 - 전체 세션 수: {len(session_module.sessions)}")
            logger.debug(f"대화 추가 전 - 세션 키 목록: {list(session_module.sessions.keys())}")
            logger.debug(f"대화 추가할 세션 ID: {session_id}")
            
            await session_module.add_conversation(
                session_id,
                chat_request.message,
                rag_result["answer"],
                {
                    "tokens_used": rag_result["tokens_used"],
                    "response_time": time.time() - start_time,
                    "sources": rag_result["sources"],
                    "topic": rag_result["topic"]
                }
            )
        
        # 4. 응답 생성
        response = ChatResponse(
            answer=rag_result["answer"],
            sources=rag_result["sources"],
            session_id=session_id,
            processing_time=time.time() - start_time,
            tokens_used=rag_result["tokens_used"],
            timestamp=datetime.now().isoformat(),
            model_info=rag_result.get("model_info")
        )
        
        # 5. 통계 업데이트
        update_stats({
            "tokens_used": rag_result["tokens_used"],
            "latency": time.time() - start_time,
            "success": True
        })
        
        # 6. 로깅 (일시적으로 비활성화)
        # await chat_logger.log_chat_request(
        #     chat_request.dict(),
        #     response.dict(),
        #     time.time() - start_time,
        #     session_id
        # )
        
        logger.info("Chat request completed successfully", 
                   session_id=session_id,
                   message_length=len(chat_request.message),
                   processing_time=time.time() - start_time,
                   tokens_used=rag_result["tokens_used"],
                   sources_count=len(rag_result["sources"]))
        
        return response
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Chat API error", error=str(error), session_id=session_id)
        
        update_stats({"success": False})
        
        # 구체적인 오류 분류 및 사용자 친화적 메시지 생성
        error_message = str(error)
        error_type = "unknown"
        user_message = "요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        if "응답 시간이 초과" in error_message or "timeout" in error_message.lower():
            error_type = "timeout"
            user_message = "AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
        elif "API" in error_message.upper() or "KEY" in error_message.upper() or "credentials" in error_message.lower():
            error_type = "api_auth"
            user_message = "AI 서비스 연결에 문제가 있습니다. 관리자에게 문의해주세요."
        elif "document" in error_message.lower() or "retrieval" in error_message.lower():
            error_type = "retrieval"
            user_message = "문서 검색 중 오류가 발생했습니다. 질문을 다시 입력해주세요."
        elif "session" in error_message.lower():
            error_type = "session"
            user_message = "세션 처리 중 오류가 발생했습니다. 새로고침 후 다시 시도해주세요."
        elif "model" in error_message.lower() or "generation" in error_message.lower():
            error_type = "generation" 
            user_message = "AI 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        # 에러 응답에 올바른 구조 보장
        error_response = ChatResponse(
            answer=user_message,
            sources=[],
            session_id=session_id or "unknown",  # session_id가 None인 경우 기본값 사용
            processing_time=time.time() - start_time,
            tokens_used=0,
            timestamp=datetime.now().isoformat(),
            model_info={
                "provider": "error",
                "model": "none",
                "generation_time": 0,
                "model_config": {
                    "error_type": error_type,
                    "error_handled": True
                }
            }
        )
        
        # 로그에 상세한 에러 정보 기록
        logger.error("Detailed error info", 
                    error_type=error_type, 
                    session_id=session_id,
                    processing_time=time.time() - start_time,
                    message_length=len(chat_request.message) if chat_request else 0)
        
        return error_response

@router.post("/chat/session", response_model=SessionResponse)
async def create_session(request: Request, session_request: SessionCreateRequest):
    """새 세션 생성"""
    try:
        context = get_request_context(request)
        context.update(session_request.metadata)
        
        session_module = modules.get('session')
        if not session_module:
            raise HTTPException(status_code=500, detail="Session module not available")
        
        new_session = await session_module.create_session({"metadata": context})
        
        logger.info(f"New session created: {new_session['session_id']}")
        
        return SessionResponse(
            session_id=new_session["session_id"],
            message="Session created successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as error:
        logger.error("Create session error", error=str(error))
        raise HTTPException(status_code=500, detail="Failed to create session")

@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, limit: int = 20, offset: int = 0):
    """채팅 히스토리 조회"""
    try:
        session_module = modules.get('session')
        if not session_module:
            raise HTTPException(status_code=500, detail="Session module not available")
        
        history = await session_module.get_chat_history(session_id)
        
        # 페이지네이션 적용
        start = offset
        end = start + limit
        paginated_messages = history["messages"][start:end]
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=paginated_messages,
            total_messages=history["message_count"],
            limit=limit,
            offset=offset,
            has_more=end < history["message_count"]
        )
        
    except Exception as error:
        logger.error("Get chat history error", error=str(error))
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        session_module = modules.get('session')
        if not session_module:
            raise HTTPException(status_code=500, detail="Session module not available")
        
        await session_module.delete_session(session_id)
        
        return {
            "message": "Session deleted successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error("Delete session error", error=str(error))
        raise HTTPException(status_code=500, detail="Failed to delete session")

@router.get("/chat/stats", response_model=StatsResponse)
async def get_stats():
    """통계 조회"""
    try:
        session_module = modules.get('session')
        session_stats = await session_module.get_stats() if session_module else {}
        
        return StatsResponse(
            chat=stats,
            session=session_stats,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as error:
        logger.error("Get stats error", error=str(error))
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
"""
Enhanced Session management module with LangChain memory
LangChain 메모리를 활용한 향상된 세션 관리 모듈
"""
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4
import json

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory

from ..lib.logger import get_logger

logger = get_logger(__name__)

class EnhancedSessionModule:
    """향상된 세션 관리 모듈 with LangChain Memory"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ttl = config.get('ttl', 3600)  # 기본 1시간
        self.max_exchanges = config.get('max_exchanges', 5)  # 최대 교환 수
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5분
        
        # LLM for summarization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=config.get('llm', {}).get('google', {}).get('api_key'),
            temperature=0.3
        )
        
        # 인메모리 세션 저장소
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.memories: Dict[str, Any] = {}  # LangChain memory objects
        
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_conversations': 0,
            'cleanup_runs': 0
        }
        
        self.cleanup_task = None
        
    async def initialize(self):
        """모듈 초기화"""
        try:
            logger.info("Initializing enhanced session module...")
            
            # 정리 작업 시작
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Enhanced session module initialized successfully")
            
        except Exception as e:
            logger.error(f"Enhanced session module initialization failed: {e}")
            raise
            
    async def destroy(self):
        """모듈 정리"""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            self.sessions.clear()
            self.memories.clear()
            logger.info("Enhanced session module destroyed")
            
        except Exception as e:
            logger.error(f"Enhanced session module destroy error: {e}")
    
    async def create_session(self, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """새 세션 생성"""
        session_id = str(uuid4())
        
        # LangChain memory 생성
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True,
            chat_memory=InMemoryChatMessageHistory(),
            memory_key="chat_history",
            input_key="human_input",
            output_key="ai_output"
        )
        
        session_data = {
            'session_id': session_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'metadata': metadata or {},
            'user_name': None,  # 사용자 이름 저장
            'user_info': {},    # 추가 사용자 정보
            'topics': [],       # 대화 주제들
            'facts': {}         # 대화에서 추출된 사실들
        }
        
        self.sessions[session_id] = session_data
        self.memories[session_id] = memory
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] += 1
        
        logger.debug(f"Enhanced session created: {session_id}")
        
        return {'session_id': session_id}
    
    async def get_session(self, session_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """세션 조회"""
        if session_id not in self.sessions:
            return {
                'is_valid': False,
                'reason': 'session_not_found'
            }
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # TTL 검사
        if current_time - session['last_accessed'] > self.ttl:
            await self.delete_session(session_id)
            return {
                'is_valid': False,
                'reason': 'session_expired'
            }
        
        # 마지막 접근 시간 업데이트
        session['last_accessed'] = current_time
        
        # 컨텍스트 정보 업데이트 (있는 경우)
        if context:
            session['metadata'].update(context)
        
        return {
            'is_valid': True,
            'session': session,
            'renewed_session_id': session_id
        }
    
    async def delete_session(self, session_id: str):
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.memories:
                del self.memories[session_id]
            self.stats['active_sessions'] = max(0, self.stats['active_sessions'] - 1)
            logger.debug(f"Enhanced session deleted: {session_id}")
    
    async def add_conversation(self, session_id: str, user_message: str, 
                             assistant_response: str, metadata: Dict[str, Any] = None):
        """대화 추가 with fact extraction"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            raise ValueError(f"Invalid session: {session_id}")
        
        session = session_result['session']
        memory = self.memories.get(session_id)
        
        if not memory:
            raise ValueError(f"Memory not found for session: {session_id}")
        
        # 사용자 정보 추출
        await self._extract_user_info(session, user_message)
        
        # LangChain memory에 대화 추가
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(assistant_response)
        
        # 메모리 저장 (summarization 트리거)
        memory.save_context(
            {"human_input": user_message},
            {"ai_output": assistant_response}
        )
        
        self.stats['total_conversations'] += 1
        
        # 토픽 추출
        if metadata and metadata.get('topic'):
            if metadata['topic'] not in session['topics']:
                session['topics'].append(metadata['topic'])
        
        logger.debug(f"Conversation added to enhanced session: {session_id}")
    
    async def get_context_string(self, session_id: str) -> str:
        """세션 컨텍스트 문자열 반환 with LangChain memory"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            return ""
        
        session = session_result['session']
        memory = self.memories.get(session_id)
        
        if not memory:
            return ""
        
        context_parts = []
        
        # 사용자 정보 추가
        if session.get('user_name'):
            context_parts.append(f"사용자 이름: {session['user_name']}")
        
        if session.get('user_info'):
            for key, value in session['user_info'].items():
                context_parts.append(f"사용자 {key}: {value}")
        
        # 대화 주제들
        if session.get('topics'):
            context_parts.append(f"대화 주제: {', '.join(session['topics'])}")
        
        # LangChain 메모리에서 대화 내역 가져오기
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get('chat_history', [])
        
        if chat_history:
            context_parts.append("\n최근 대화 내역:")
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    context_parts.append(f"사용자: {message.content}")
                elif isinstance(message, AIMessage):
                    context_parts.append(f"AI: {message.content}")
        
        # 중요 사실들
        if session.get('facts'):
            context_parts.append("\n기억된 정보:")
            for key, value in session['facts'].items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    async def get_chat_history(self, session_id: str) -> Dict[str, Any]:
        """채팅 히스토리 반환"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            return {
                'messages': [],
                'message_count': 0
            }
        
        memory = self.memories.get(session_id)
        if not memory:
            return {
                'messages': [],
                'message_count': 0
            }
        
        messages = []
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get('chat_history', [])
        
        for i, message in enumerate(chat_history):
            msg_data = {
                'type': 'user' if isinstance(message, HumanMessage) else 'assistant',
                'content': message.content,
                'timestamp': datetime.now().isoformat()  # 실제로는 저장된 타임스탬프 사용
            }
            messages.append(msg_data)
        
        return {
            'messages': messages,
            'message_count': len(messages)
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        current_time = time.time()
        
        # 활성 세션 재계산
        active_count = 0
        for session in self.sessions.values():
            if current_time - session['last_accessed'] <= self.ttl:
                active_count += 1
        
        self.stats['active_sessions'] = active_count
        
        return {
            **self.stats,
            'total_sessions_in_memory': len(self.sessions),
            'ttl_seconds': self.ttl,
            'max_exchanges': self.max_exchanges
        }
    
    async def clear_cache(self):
        """캐시 클리어"""
        expired_sessions = []
        current_time = time.time()
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_accessed'] > self.ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        logger.info(f"Cache cleared: {len(expired_sessions)} expired sessions removed")
    
    async def _cleanup_loop(self):
        """정리 작업 루프"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session['last_accessed'] > self.ttl:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.delete_session(session_id)
                
                if expired_sessions:
                    logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                self.stats['cleanup_runs'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def _extract_user_info(self, session: Dict[str, Any], message: str):
        """메시지에서 사용자 정보 추출"""
        # 이름 추출
        name_patterns = [
            "내 이름은 ",
            "저는 ",
            "제 이름은 ",
            "나는 ",
        ]
        
        for pattern in name_patterns:
            if pattern in message:
                parts = message.split(pattern)
                if len(parts) > 1:
                    name_candidate = parts[1].split()[0].rstrip('이야입니다요.')
                    if name_candidate and len(name_candidate) < 10:  # 합리적인 이름 길이
                        session['user_name'] = name_candidate
                        session['facts']['이름'] = name_candidate
                        logger.info(f"Extracted user name: {name_candidate}")
                        break
        
        # 기타 정보 추출 (나이, 직업 등)
        if "살" in message and any(char.isdigit() for char in message):
            # 나이 추출 로직
            import re
            age_match = re.search(r'(\d+)\s*살', message)
            if age_match:
                age = int(age_match.group(1))
                if 1 < age < 120:  # 합리적인 나이 범위
                    session['user_info']['나이'] = age
                    session['facts']['나이'] = f"{age}살"
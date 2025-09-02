"""
Session management module
세션 관리 및 컨텍스트 처리 모듈
"""
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4
import json

from ..lib.logger import get_logger

logger = get_logger(__name__)

class SessionModule:
    """세션 관리 모듈"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ttl = config.get('ttl', 3600)  # 기본 1시간
        self.max_context_length = config.get('max_context_length', 5)
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5분
        
        # 인메모리 세션 저장소 (실제 운영에서는 Redis 사용)
        self.sessions: Dict[str, Dict[str, Any]] = {}
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
            logger.info("Initializing session module...")
            
            # 정리 작업 시작
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Session module initialized successfully")
            
        except Exception as e:
            logger.error(f"Session module initialization failed: {e}")
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
            logger.info("Session module destroyed")
            
        except Exception as e:
            logger.error(f"Session module destroy error: {e}")
    
    async def create_session(self, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """새 세션 생성"""
        session_id = str(uuid4())
        
        session_data = {
            'session_id': session_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'conversations': [],
            'metadata': metadata or {},
            'context_summary': None
        }
        
        self.sessions[session_id] = session_data
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] += 1
        
        logger.debug(f"Session created: {session_id}")
        
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
            self.stats['active_sessions'] = max(0, self.stats['active_sessions'] - 1)
            logger.debug(f"Session deleted: {session_id}")
    
    async def add_conversation(self, session_id: str, user_message: str, 
                             assistant_response: str, metadata: Dict[str, Any] = None):
        """대화 추가"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            raise ValueError(f"Invalid session: {session_id}")
        
        session = session_result['session']
        
        conversation = {
            'timestamp': time.time(),
            'user_message': user_message,
            'assistant_response': assistant_response,
            'metadata': metadata or {}
        }
        
        session['conversations'].append(conversation)
        self.stats['total_conversations'] += 1
        
        # 컨텍스트 길이 제한
        if len(session['conversations']) > self.max_context_length:
            # 오래된 대화 제거하되, 요약 생성
            old_conversations = session['conversations'][:-self.max_context_length]
            session['conversations'] = session['conversations'][-self.max_context_length:]
            
            # 컨텍스트 요약 업데이트
            session['context_summary'] = await self._summarize_conversations(old_conversations)
        
        logger.debug(f"Conversation added to session: {session_id}")
    
    async def get_context_string(self, session_id: str) -> str:
        """세션 컨텍스트 문자열 반환"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            return ""
        
        session = session_result['session']
        context_parts = []
        
        # 요약된 컨텍스트 추가
        if session.get('context_summary'):
            context_parts.append(f"이전 대화 요약: {session['context_summary']}")
        
        # 최근 대화 추가
        for conv in session['conversations'][-3:]:  # 최근 3개 대화만
            context_parts.append(f"사용자: {conv['user_message']}")
            context_parts.append(f"어시스턴트: {conv['assistant_response']}")
        
        return "\n".join(context_parts)
    
    async def get_chat_history(self, session_id: str) -> Dict[str, Any]:
        """채팅 히스토리 반환"""
        session_result = await self.get_session(session_id)
        if not session_result['is_valid']:
            return {
                'messages': [],
                'message_count': 0
            }
        
        session = session_result['session']
        messages = []
        
        for conv in session['conversations']:
            messages.extend([
                {
                    'type': 'user',
                    'content': conv['user_message'],
                    'timestamp': datetime.fromtimestamp(conv['timestamp']).isoformat()
                },
                {
                    'type': 'assistant',
                    'content': conv['assistant_response'],
                    'timestamp': datetime.fromtimestamp(conv['timestamp']).isoformat(),
                    'metadata': conv['metadata']
                }
            ])
        
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
            'max_context_length': self.max_context_length
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
    
    async def _summarize_conversations(self, conversations: List[Dict[str, Any]]) -> str:
        """대화 요약 생성"""
        # 간단한 요약 로직 (실제로는 LLM을 사용할 수 있음)
        if not conversations:
            return ""
        
        topics = []
        for conv in conversations:
            user_msg = conv['user_message']
            # 간단한 키워드 추출
            if any(word in user_msg.lower() for word in ['검색', '찾기', '찾아']):
                topics.append('문서 검색')
            elif any(word in user_msg.lower() for word in ['설명', '알려', '도움']):
                topics.append('정보 요청')
            elif any(word in user_msg.lower() for word in ['분석', '분석해']):
                topics.append('내용 분석')
        
        if topics:
            unique_topics = list(set(topics))
            return f"이전에 {', '.join(unique_topics)} 관련 대화를 나누었습니다."
        else:
            return f"이전에 {len(conversations)}개의 대화를 나누었습니다."
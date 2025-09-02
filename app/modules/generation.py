"""
Generation module
Multi-LLM 답변 생성 모듈 (Gemini, Claude, OpenAI 지원)
"""
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

# AI clients
import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI

from ..lib.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GenerationResult:
    """생성 결과 데이터 클래스"""
    answer: str
    text: str  # 하위 호환성
    tokens_used: int
    model_used: str
    provider: str
    generation_time: float
    model_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # text는 answer의 alias
        if not self.text:
            self.text = self.answer

class GenerationModule:
    """답변 생성 모듈"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # config.yaml의 llm 섹션에서 직접 가져오기
        self.llm_config = config.get('llm', {})
        
        # 클라이언트 저장소
        self.clients = {}
        
        # 기본 설정
        self.default_provider = self.llm_config.get('default_provider', 'google')
        self.auto_fallback = self.llm_config.get('auto_fallback', True)
        self.fallback_order = self.llm_config.get('fallback_order', ['google', 'openai', 'anthropic'])
        
        # 프롬프트 설정
        self.prompts = config.get('prompts', {})
        
        # 통계
        self.stats = {
            'total_generations': 0,
            'generations_by_provider': {},
            'total_tokens': 0,
            'average_generation_time': 0.0,
            'fallback_count': 0,
            'error_count': 0
        }
    
    async def initialize(self):
        """모듈 초기화"""
        try:
            logger.info("Initializing generation module...")
            
            # providers는 list이므로, 각 provider의 설정은 llm_config에서 직접 가져옴
            providers_list = self.llm_config.get('providers', [])
            
            # Google Gemini 클라이언트
            if 'google' in providers_list and 'google' in self.llm_config:
                await self._init_gemini(self.llm_config['google'])
            
            # OpenAI 클라이언트  
            if 'openai' in providers_list and 'openai' in self.llm_config:
                await self._init_openai(self.llm_config['openai'])
            
            # Anthropic Claude 클라이언트
            if 'anthropic' in providers_list and 'anthropic' in self.llm_config:
                await self._init_claude(self.llm_config['anthropic'])
            
            if not self.clients:
                raise ValueError("No LLM providers configured")
            
            logger.info(f"Generation module initialized with providers: {list(self.clients.keys())}")
            
        except Exception as e:
            logger.error(f"Generation module initialization failed: {e}")
            raise
    
    async def destroy(self):
        """모듈 정리"""
        try:
            self.clients.clear()
            logger.info("Generation module destroyed")
        except Exception as e:
            logger.error(f"Generation module destroy error: {e}")
    
    async def _init_gemini(self, config: Dict[str, Any]):
        """Gemini 클라이언트 초기화"""
        try:
            api_key = config.get('api_key')
            if not api_key:
                logger.warning("Gemini API key not found, skipping initialization")
                return
            
            genai.configure(api_key=api_key)
            
            model_name = config.get('model', 'gemini-2.0-flash-exp')
            model = genai.GenerativeModel(model_name)
            
            self.clients['google'] = {
                'client': model,
                'config': config,
                'model': model_name
            }
            
            logger.info(f"Gemini client initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
    
    async def _init_openai(self, config: Dict[str, Any]):
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = config.get('api_key')
            if not api_key:
                logger.warning("OpenAI API key not found, skipping initialization")
                return
            
            client = OpenAI(api_key=api_key)
            
            model_name = config.get('model', 'gpt-4o-mini')
            
            self.clients['openai'] = {
                'client': client,
                'config': config,
                'model': model_name
            }
            
            logger.info(f"OpenAI client initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
    
    async def _init_claude(self, config: Dict[str, Any]):
        """Claude 클라이언트 초기화"""
        try:
            api_key = config.get('api_key')
            if not api_key:
                logger.warning("Claude API key not found, skipping initialization")
                return
            
            client = Anthropic(api_key=api_key)
            
            model_name = config.get('model', 'claude-3-5-haiku-20241022')
            
            self.clients['anthropic'] = {
                'client': client,
                'config': config,
                'model': model_name
            }
            
            logger.info(f"Claude client initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Claude initialization failed: {e}")
    
    async def generate_answer(self, query: str, context_documents: List[Any], 
                            options: Dict[str, Any] = None) -> GenerationResult:
        """답변 생성 (메인 메서드)"""
        start_time = asyncio.get_event_loop().time()
        options = options or {}
        
        self.stats['total_generations'] += 1
        
        # 프로바이더 결정
        provider = options.get('provider', self.default_provider)
        
        # 폴백 시나리오
        providers_to_try = [provider]
        if self.auto_fallback and provider in self.fallback_order:
            # 기본 프로바이더 이후의 프로바이더들을 폴백으로 추가
            current_index = self.fallback_order.index(provider)
            providers_to_try.extend(self.fallback_order[current_index + 1:])
        
        last_error = None
        
        for attempt_provider in providers_to_try:
            if attempt_provider not in self.clients:
                logger.warning(f"Provider {attempt_provider} not available, skipping")
                continue
            
            try:
                logger.debug(f"Attempting generation with {attempt_provider}")
                
                result = await self._generate_with_provider(
                    attempt_provider, query, context_documents, options
                )
                
                # 생성 시간 계산
                generation_time = asyncio.get_event_loop().time() - start_time
                result.generation_time = generation_time
                
                # 통계 업데이트
                self._update_stats(attempt_provider, result.tokens_used, generation_time)
                
                if attempt_provider != provider:
                    self.stats['fallback_count'] += 1
                    logger.info(f"Successfully fell back to {attempt_provider}")
                
                return result
                
            except Exception as e:
                logger.error(f"Generation failed with {attempt_provider}: {e}")
                last_error = e
                continue
        
        # 모든 프로바이더 실패
        self.stats['error_count'] += 1
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    async def _generate_with_provider(self, provider: str, query: str, 
                                    context_documents: List[Any], options: Dict[str, Any]) -> GenerationResult:
        """특정 프로바이더로 생성"""
        client_info = self.clients[provider]
        
        # 컨텍스트 구성
        context_text = self._build_context(context_documents)
        
        # 프롬프트 구성
        prompt = self._build_prompt(query, context_text, options)
        
        # 프로바이더별 생성
        if provider == 'google':
            return await self._generate_gemini(client_info, prompt, options)
        elif provider == 'openai':
            return await self._generate_openai(client_info, prompt, options)
        elif provider == 'anthropic':
            return await self._generate_claude(client_info, prompt, options)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _generate_gemini(self, client_info: Dict[str, Any], prompt: str, 
                             options: Dict[str, Any]) -> GenerationResult:
        """Gemini로 생성"""
        try:
            model = client_info['client']
            config = client_info['config']
            
            # 생성 설정
            generation_config = {
                'temperature': config.get('temperature', 0.7),
                'max_output_tokens': options.get('max_tokens', 2000),
                'top_p': config.get('top_p', 0.95),
                'top_k': config.get('top_k', 40)
            }
            
            # 생성 실행
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(**generation_config)
            )
            
            # 토큰 사용량 추정 (Gemini는 정확한 토큰 카운트를 제공하지 않을 수 있음)
            tokens_used = len(response.text.split()) * 1.3  # 대략적인 추정
            tokens_used = int(tokens_used)
            
            return GenerationResult(
                answer=response.text,
                text=response.text,
                tokens_used=tokens_used,
                model_used=client_info['model'],
                provider='google',
                generation_time=0,  # 나중에 설정됨
                model_config=generation_config
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def _generate_openai(self, client_info: Dict[str, Any], prompt: str, 
                             options: Dict[str, Any]) -> GenerationResult:
        """OpenAI로 생성"""
        try:
            client = client_info['client']
            config = client_info['config']
            
            # 메시지 구성
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 생성 설정
            generation_config = {
                'temperature': config.get('temperature', 0.7),
                'max_tokens': options.get('max_tokens', 2000),
                'top_p': config.get('top_p', 1.0)
            }
            
            # 생성 실행
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=client_info['model'],
                messages=messages,
                **generation_config
            )
            
            # 결과 추출
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return GenerationResult(
                answer=answer,
                text=answer,
                tokens_used=tokens_used,
                model_used=client_info['model'],
                provider='openai',
                generation_time=0,  # 나중에 설정됨
                model_config=generation_config
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def _generate_claude(self, client_info: Dict[str, Any], prompt: str, 
                             options: Dict[str, Any]) -> GenerationResult:
        """Claude로 생성"""
        try:
            client = client_info['client']
            config = client_info['config']
            
            # 생성 설정
            generation_config = {
                'model': client_info['model'],
                'max_tokens': options.get('max_tokens', 2000),
                'temperature': config.get('temperature', 0.7)
            }
            
            # 생성 실행
            response = await asyncio.to_thread(
                client.messages.create,
                **generation_config,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 결과 추출
            answer = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return GenerationResult(
                answer=answer,
                text=answer,
                tokens_used=tokens_used,
                model_used=client_info['model'],
                provider='anthropic',
                generation_time=0,  # 나중에 설정됨
                model_config=generation_config
            )
            
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise
    
    def _build_context(self, context_documents: List[Any]) -> str:
        """컨텍스트 텍스트 구성"""
        if not context_documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(context_documents[:10]):  # 최대 10개 문서
            # 문서 내용 추출 (다양한 형식 지원)
            content = ""
            if hasattr(doc, 'content'):
                content = doc.content
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get('content', '')
            elif isinstance(doc, str):
                content = doc
            
            if content:
                context_parts.append(f"[문서 {i+1}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context_text: str, options: Dict[str, Any]) -> str:
        """프롬프트 구성"""
        style = options.get('style', 'standard')
        session_context = options.get('session_context', '')
        
        # 기본 시스템 프롬프트
        system_prompt = self.prompts.get('system', """
당신은 한국어로 답변하는 도움이 되는 AI 어시스턴트입니다.
제공된 문서 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.
""").strip()
        
        # 스타일별 프롬프트 조정
        if style == 'detailed':
            system_prompt += "\n자세하고 포괄적인 답변을 제공해주세요."
        elif style == 'concise':
            system_prompt += "\n간결하고 요점만 정리한 답변을 제공해주세요."
        
        # 프롬프트 조합
        prompt_parts = [system_prompt]
        
        if session_context:
            prompt_parts.append(f"\n이전 대화 맥락:\n{session_context}\n")
        
        if context_text:
            prompt_parts.append(f"\n참고 문서:\n{context_text}\n")
        
        prompt_parts.append(f"\n사용자 질문: {query}\n")
        prompt_parts.append("\n답변:")
        
        return "\n".join(prompt_parts)
    
    def _update_stats(self, provider: str, tokens_used: int, generation_time: float):
        """통계 업데이트"""
        # 프로바이더별 통계
        if provider not in self.stats['generations_by_provider']:
            self.stats['generations_by_provider'][provider] = 0
        self.stats['generations_by_provider'][provider] += 1
        
        # 토큰 통계
        self.stats['total_tokens'] += tokens_used
        
        # 평균 생성 시간 업데이트
        current_avg = self.stats['average_generation_time']
        total_gens = self.stats['total_generations']
        self.stats['average_generation_time'] = (
            (current_avg * (total_gens - 1) + generation_time) / total_gens
        )
    
    async def get_available_providers(self) -> List[str]:
        """사용 가능한 프로바이더 목록"""
        return list(self.clients.keys())
    
    async def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            'available_providers': list(self.clients.keys()),
            'default_provider': self.default_provider,
            'auto_fallback': self.auto_fallback,
            'fallback_order': self.fallback_order
        }
    
    async def test_provider(self, provider: str) -> Dict[str, Any]:
        """프로바이더 테스트"""
        if provider not in self.clients:
            return {
                'success': False,
                'error': f'Provider {provider} not available'
            }
        
        try:
            # 간단한 테스트 생성
            result = await self._generate_with_provider(
                provider, 
                "안녕하세요", 
                [], 
                {'max_tokens': 50}
            )
            
            return {
                'success': True,
                'model': result.model_used,
                'response_length': len(result.answer),
                'tokens_used': result.tokens_used
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
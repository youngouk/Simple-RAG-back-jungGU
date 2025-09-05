"""
프롬프트 관리 모듈
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from ..lib.logger import get_logger
from ..models.prompts import PromptCreate, PromptUpdate, PromptResponse

logger = get_logger(__name__)

class PromptManager:
    """프롬프트 관리 클래스"""
    
    def __init__(self, storage_path: str = "./data/prompts"):
        """
        프롬프트 매니저 초기화
        
        Args:
            storage_path: 프롬프트 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.prompts_file = self.storage_path / "prompts.json"
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self._load_prompts()
        self._ensure_default_prompts()
    
    def _load_prompts(self):
        """저장된 프롬프트 로드"""
        try:
            if self.prompts_file.exists():
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.prompts = data.get('prompts', {})
                logger.info(f"Loaded {len(self.prompts)} prompts from storage")
            else:
                logger.info("No existing prompts file found, starting fresh")
                self.prompts = {}
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.prompts = {}
    
    def _save_prompts(self):
        """프롬프트 저장"""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump({'prompts': self.prompts}, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.prompts)} prompts to storage")
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
            raise
    
    def _ensure_default_prompts(self):
        """기본 프롬프트 확인 및 생성"""
        default_prompts = [
            {
                "name": "system",
                "content": """당신은 유저의 질문을 분석/판단하고, 질문에 부합하는 정보를 content 내에서 찾아 한국어로 답변하는 AI 어시스턴트입니다.
제공된 문서 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.""",
                "description": "기본 시스템 프롬프트",
                "category": "system",
                "is_active": True
            },
            {
                "name": "detailed",
                "content": """당신은 유저의 질문을 분석/판단하고, 질문에 부합하는 정보를 content 내에서 찾아 한국어로 답변하는 AI 어시스턴트입니다.
제공된 문서 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.
자세하고 포괄적인 답변을 제공해주세요. 관련된 모든 정보를 포함하여 설명해주세요.""",
                "description": "자세한 답변 스타일 프롬프트",
                "category": "style",
                "is_active": True
            },
            {
                "name": "concise",
                "content": """당신은 유저의 질문을 분석/판단하고, 질문에 부합하는 정보를 content 내에서 찾아 한국어로 답변하는 AI 어시스턴트입니다.
제공된 문서 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.
간결하고 요점만 정리한 답변을 제공해주세요. 핵심만 명확하게 전달해주세요.""",
                "description": "간결한 답변 스타일 프롬프트",
                "category": "style",
                "is_active": True
            },
            {
                "name": "professional",
                "content": """당신은 전문적인 비즈니스 컨설턴트입니다.
제공된 문서 정보를 바탕으로 전문적이고 신뢰할 수 있는 답변을 제공해주세요.
업계 표준과 베스트 프랙티스를 고려하여 답변해주세요.""",
                "description": "전문적 답변 스타일 프롬프트",
                "category": "style", 
                "is_active": True
            },
            {
                "name": "educational",
                "content": """당신은 친절한 교육자입니다.
제공된 문서 정보를 바탕으로 이해하기 쉽게 설명해주세요.
필요한 경우 예시를 들어 설명하고, 단계별로 차근차근 안내해주세요.""",
                "description": "교육적 답변 스타일 프롬프트",
                "category": "style",
                "is_active": True
            }
        ]
        
        # 기본 프롬프트 중 없는 것만 추가
        added_count = 0
        for prompt_data in default_prompts:
            # name으로 기존 프롬프트 검색
            existing = None
            for pid, p in self.prompts.items():
                if p.get('name') == prompt_data['name']:
                    existing = pid
                    break
            
            if not existing:
                prompt_id = str(uuid.uuid4())
                self.prompts[prompt_id] = {
                    **prompt_data,
                    'id': prompt_id,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'metadata': {}
                }
                added_count += 1
        
        if added_count > 0:
            self._save_prompts()
            logger.info(f"Added {added_count} default prompts")
    
    async def get_prompt(self, prompt_id: Optional[str] = None, name: Optional[str] = None) -> Optional[PromptResponse]:
        """
        프롬프트 조회
        
        Args:
            prompt_id: 프롬프트 ID
            name: 프롬프트 이름
            
        Returns:
            프롬프트 정보 또는 None
        """
        try:
            # ID로 조회
            if prompt_id and prompt_id in self.prompts:
                prompt_data = self.prompts[prompt_id]
                return PromptResponse(**prompt_data)
            
            # 이름으로 조회
            if name:
                for pid, prompt_data in self.prompts.items():
                    if prompt_data.get('name') == name and prompt_data.get('is_active', True):
                        return PromptResponse(**prompt_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            return None
    
    async def get_prompt_content(self, name: str, default: Optional[str] = None) -> str:
        """
        프롬프트 내용만 조회 (generation.py에서 사용)
        
        Args:
            name: 프롬프트 이름
            default: 기본값
            
        Returns:
            프롬프트 내용
        """
        prompt = await self.get_prompt(name=name)
        if prompt and prompt.is_active:
            return prompt.content
        return default or ""
    
    async def list_prompts(self, category: Optional[str] = None, is_active: Optional[bool] = None,
                          page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """
        프롬프트 목록 조회
        
        Args:
            category: 카테고리 필터
            is_active: 활성화 상태 필터
            page: 페이지 번호
            page_size: 페이지 크기
            
        Returns:
            프롬프트 목록
        """
        try:
            # 필터링
            filtered_prompts = []
            for prompt_id, prompt_data in self.prompts.items():
                if category and prompt_data.get('category') != category:
                    continue
                if is_active is not None and prompt_data.get('is_active') != is_active:
                    continue
                filtered_prompts.append(PromptResponse(**prompt_data))
            
            # 정렬 (최근 수정 순)
            filtered_prompts.sort(key=lambda x: x.updated_at, reverse=True)
            
            # 페이지네이션
            total = len(filtered_prompts)
            start = (page - 1) * page_size
            end = start + page_size
            paginated = filtered_prompts[start:end]
            
            return {
                'prompts': paginated,
                'total': total,
                'page': page,
                'page_size': page_size
            }
            
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return {
                'prompts': [],
                'total': 0,
                'page': page,
                'page_size': page_size
            }
    
    async def create_prompt(self, prompt_data: PromptCreate) -> PromptResponse:
        """
        프롬프트 생성
        
        Args:
            prompt_data: 프롬프트 생성 데이터
            
        Returns:
            생성된 프롬프트
        """
        try:
            # 중복 이름 체크
            for pid, p in self.prompts.items():
                if p.get('name') == prompt_data.name:
                    raise ValueError(f"Prompt with name '{prompt_data.name}' already exists")
            
            prompt_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            prompt = {
                'id': prompt_id,
                **prompt_data.model_dump(),
                'created_at': now,
                'updated_at': now
            }
            
            self.prompts[prompt_id] = prompt
            self._save_prompts()
            
            logger.info(f"Created prompt: {prompt_id} ({prompt_data.name})")
            return PromptResponse(**prompt)
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            raise
    
    async def update_prompt(self, prompt_id: str, update_data: PromptUpdate) -> PromptResponse:
        """
        프롬프트 업데이트
        
        Args:
            prompt_id: 프롬프트 ID
            update_data: 업데이트 데이터
            
        Returns:
            업데이트된 프롬프트
        """
        try:
            if prompt_id not in self.prompts:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # 이름 중복 체크 (다른 프롬프트와)
            if update_data.name:
                for pid, p in self.prompts.items():
                    if pid != prompt_id and p.get('name') == update_data.name:
                        raise ValueError(f"Prompt with name '{update_data.name}' already exists")
            
            # 업데이트
            prompt = self.prompts[prompt_id]
            update_dict = update_data.model_dump(exclude_unset=True)
            
            for key, value in update_dict.items():
                if value is not None:
                    prompt[key] = value
            
            prompt['updated_at'] = datetime.now().isoformat()
            
            self._save_prompts()
            
            logger.info(f"Updated prompt: {prompt_id}")
            return PromptResponse(**prompt)
            
        except Exception as e:
            logger.error(f"Error updating prompt: {e}")
            raise
    
    async def delete_prompt(self, prompt_id: str) -> bool:
        """
        프롬프트 삭제
        
        Args:
            prompt_id: 프롬프트 ID
            
        Returns:
            성공 여부
        """
        try:
            if prompt_id not in self.prompts:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # 기본 시스템 프롬프트는 삭제 불가
            prompt = self.prompts[prompt_id]
            if prompt.get('name') == 'system' and prompt.get('category') == 'system':
                raise ValueError("Cannot delete default system prompt")
            
            del self.prompts[prompt_id]
            self._save_prompts()
            
            logger.info(f"Deleted prompt: {prompt_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting prompt: {e}")
            raise
    
    async def export_prompts(self) -> Dict[str, Any]:
        """
        모든 프롬프트 내보내기
        
        Returns:
            프롬프트 데이터
        """
        try:
            return {
                'prompts': [PromptResponse(**p).model_dump() for p in self.prompts.values()],
                'exported_at': datetime.now().isoformat(),
                'total': len(self.prompts)
            }
        except Exception as e:
            logger.error(f"Error exporting prompts: {e}")
            raise
    
    async def import_prompts(self, data: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """
        프롬프트 가져오기
        
        Args:
            data: 가져올 프롬프트 데이터
            overwrite: 기존 프롬프트 덮어쓰기 여부
            
        Returns:
            가져오기 결과
        """
        try:
            imported_prompts = data.get('prompts', [])
            imported = 0
            skipped = 0
            
            for prompt_data in imported_prompts:
                # ID 재생성
                new_id = str(uuid.uuid4())
                prompt_name = prompt_data.get('name')
                
                # 이름 중복 체크
                existing_id = None
                for pid, p in self.prompts.items():
                    if p.get('name') == prompt_name:
                        existing_id = pid
                        break
                
                if existing_id and not overwrite:
                    skipped += 1
                    continue
                
                if existing_id:
                    # 덮어쓰기
                    self.prompts[existing_id].update(prompt_data)
                    self.prompts[existing_id]['updated_at'] = datetime.now().isoformat()
                else:
                    # 새로 추가
                    prompt_data['id'] = new_id
                    prompt_data['created_at'] = datetime.now().isoformat()
                    prompt_data['updated_at'] = datetime.now().isoformat()
                    self.prompts[new_id] = prompt_data
                
                imported += 1
            
            self._save_prompts()
            
            return {
                'imported': imported,
                'skipped': skipped,
                'total': len(imported_prompts)
            }
            
        except Exception as e:
            logger.error(f"Error importing prompts: {e}")
            raise

# 전역 인스턴스
_prompt_manager: Optional[PromptManager] = None

def get_prompt_manager() -> PromptManager:
    """프롬프트 매니저 인스턴스 반환"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
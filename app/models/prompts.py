"""
프롬프트 관리 데이터 모델
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class PromptBase(BaseModel):
    """프롬프트 기본 모델"""
    name: str = Field(..., description="프롬프트 이름 (예: system, detailed, concise)")
    content: str = Field(..., description="프롬프트 내용")
    description: Optional[str] = Field(None, description="프롬프트 설명")
    is_active: bool = Field(default=True, description="활성화 여부")
    category: Optional[str] = Field(default="system", description="프롬프트 카테고리")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타데이터")

class PromptCreate(PromptBase):
    """프롬프트 생성 모델"""
    pass

class PromptUpdate(BaseModel):
    """프롬프트 업데이트 모델"""
    name: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptResponse(PromptBase):
    """프롬프트 응답 모델"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="프롬프트 고유 ID")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

class PromptListResponse(BaseModel):
    """프롬프트 목록 응답 모델"""
    prompts: list[PromptResponse]
    total: int
    page: int = 1
    page_size: int = 50
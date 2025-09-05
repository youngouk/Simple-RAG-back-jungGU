"""
프롬프트 관리 API 엔드포인트
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from ..models.prompts import (
    PromptCreate, 
    PromptUpdate, 
    PromptResponse, 
    PromptListResponse
)
from ..modules.prompt_manager import get_prompt_manager
from ..lib.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

@router.get("", response_model=PromptListResponse)
@router.get("/", response_model=PromptListResponse)
async def list_prompts(
    category: Optional[str] = Query(None, description="카테고리 필터"),
    is_active: Optional[bool] = Query(None, description="활성화 상태 필터"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(50, ge=1, le=100, description="페이지 크기")
):
    """
    프롬프트 목록 조회
    
    - **category**: 카테고리로 필터링 (system, style 등)
    - **is_active**: 활성화 상태로 필터링
    - **page**: 페이지 번호 (기본값: 1)
    - **page_size**: 페이지 크기 (기본값: 50, 최대: 100)
    """
    try:
        prompt_manager = get_prompt_manager()
        result = await prompt_manager.list_prompts(
            category=category,
            is_active=is_active,
            page=page,
            page_size=page_size
        )
        return PromptListResponse(**result)
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """
    특정 프롬프트 조회
    
    - **prompt_id**: 프롬프트 고유 ID
    """
    try:
        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.get_prompt(prompt_id=prompt_id)
        
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/by-name/{name}", response_model=PromptResponse)
async def get_prompt_by_name(name: str):
    """
    이름으로 프롬프트 조회
    
    - **name**: 프롬프트 이름
    """
    try:
        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.get_prompt(name=name)
        
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt with name '{name}' not found"
            )
        
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt by name: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("", response_model=PromptResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=PromptResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt(prompt_data: PromptCreate):
    """
    새 프롬프트 생성
    
    - **name**: 프롬프트 이름 (고유해야 함)
    - **content**: 프롬프트 내용
    - **description**: 프롬프트 설명 (선택사항)
    - **category**: 카테고리 (기본값: system)
    - **is_active**: 활성화 여부 (기본값: true)
    """
    try:
        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.create_prompt(prompt_data)
        return prompt
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, update_data: PromptUpdate):
    """
    프롬프트 업데이트
    
    - **prompt_id**: 프롬프트 고유 ID
    - 업데이트할 필드만 전달하면 됨
    """
    try:
        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.update_prompt(prompt_id, update_data)
        return prompt
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(prompt_id: str):
    """
    프롬프트 삭제
    
    - **prompt_id**: 프롬프트 고유 ID
    - 기본 시스템 프롬프트는 삭제할 수 없음
    """
    try:
        prompt_manager = get_prompt_manager()
        await prompt_manager.delete_prompt(prompt_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/export/all")
async def export_prompts():
    """
    모든 프롬프트 내보내기 (백업용)
    """
    try:
        prompt_manager = get_prompt_manager()
        data = await prompt_manager.export_prompts()
        return data
    except Exception as e:
        logger.error(f"Error exporting prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/import")
async def import_prompts(
    data: dict,
    overwrite: bool = Query(False, description="기존 프롬프트 덮어쓰기 여부")
):
    """
    프롬프트 가져오기 (복원용)
    
    - **data**: export_prompts로 내보낸 데이터
    - **overwrite**: 동일한 이름의 프롬프트가 있을 경우 덮어쓸지 여부
    """
    try:
        prompt_manager = get_prompt_manager()
        result = await prompt_manager.import_prompts(data, overwrite=overwrite)
        return result
    except Exception as e:
        logger.error(f"Error importing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
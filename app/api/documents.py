"""
Documents management API endpoints
문서 관리 API 엔드포인트 - 전체 문서 일괄 삭제 포함
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..lib.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Dependencies (will be injected from main.py)
modules: Dict[str, Any] = {}
config: Dict[str, Any] = {}

def set_dependencies(app_modules: Dict[str, Any], app_config: Dict[str, Any]):
    """의존성 주입"""
    global modules, config
    modules = app_modules
    config = app_config

class BulkDeleteAllRequest(BaseModel):
    """전체 문서 일괄 삭제 요청 모델"""
    confirm_code: str = "DELETE_ALL_DOCUMENTS"
    reason: Optional[str] = None

class BulkDeleteAllResponse(BaseModel):
    """전체 문서 일괄 삭제 응답 모델"""
    deleted_count: int
    collection_cleared: bool
    operation_time_seconds: float
    message: str
    timestamp: str

class DocumentStats(BaseModel):
    """문서 통계 모델"""
    total_documents: int
    total_vectors: int
    collection_size_mb: Optional[float] = None
    oldest_document: Optional[str] = None
    newest_document: Optional[str] = None

@router.get("/documents/stats", response_model=DocumentStats)
async def get_document_stats():
    """문서 통계 조회"""
    try:
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")

        # 문서 통계 수집
        stats = await retrieval_module.get_stats()
        
        # 추가 상세 정보 조회
        collection_info = await retrieval_module.get_collection_info()
        
        return DocumentStats(
            total_documents=stats.get('total_documents', 0),
            total_vectors=stats.get('vector_count', 0),
            collection_size_mb=collection_info.get('size_mb'),
            oldest_document=collection_info.get('oldest_document'),
            newest_document=collection_info.get('newest_document')
        )
        
    except Exception as error:
        logger.error(f"Document stats error: {error}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document statistics")

@router.delete("/documents/all", response_model=BulkDeleteAllResponse)
async def delete_all_documents(
    request: BulkDeleteAllRequest,
    dry_run: bool = Query(False, description="실제 삭제 없이 시뮬레이션만 수행")
):
    """
    전체 문서 일괄 삭제
    
    ⚠️ 위험한 작업: Qdrant DB의 모든 문서와 벡터를 완전히 삭제합니다.
    
    Args:
        request: 삭제 확인 코드와 사유
        dry_run: True일 경우 실제 삭제 없이 시뮬레이션만 수행
    
    Returns:
        삭제 결과 정보
    """
    try:
        start_time = datetime.now()
        
        # 1. 확인 코드 검증
        if request.confirm_code != "DELETE_ALL_DOCUMENTS":
            raise HTTPException(
                status_code=400, 
                detail="Invalid confirmation code. Use 'DELETE_ALL_DOCUMENTS' to confirm deletion."
            )
        
        # 2. Retrieval 모듈 확인
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        # 3. 삭제 전 현재 상태 확인
        stats_before = await retrieval_module.get_stats()
        total_documents = stats_before.get('total_documents', 0)
        total_vectors = stats_before.get('vector_count', 0)
        
        if total_documents == 0:
            return BulkDeleteAllResponse(
                deleted_count=0,
                collection_cleared=True,
                operation_time_seconds=0.0,
                message="No documents found. Collection is already empty.",
                timestamp=datetime.now().isoformat()
            )
        
        # 4. Dry run 모드
        if dry_run:
            logger.info(f"DRY RUN - Would delete {total_documents} documents, {total_vectors} vectors")
            
            return BulkDeleteAllResponse(
                deleted_count=total_documents,
                collection_cleared=True,
                operation_time_seconds=0.0,
                message=f"DRY RUN - Would delete {total_documents} documents and {total_vectors} vectors",
                timestamp=datetime.now().isoformat()
            )
        
        # 5. 실제 삭제 실행
        logger.warning(f"BULK DELETE ALL initiated: {total_documents} documents, reason: {request.reason}")
        
        try:
            # Qdrant 컬렉션 전체 삭제 또는 모든 포인트 삭제
            collection_cleared = await retrieval_module.delete_all_documents()
            
        except Exception as deletion_error:
            operation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Deletion process failed: {deletion_error}")
            
            # 삭제 프로세스 중 오류 발생, 하지만 실제 문서는 삭제되었을 수 있음
            # 현재 상태 재확인
            try:
                stats_after = await retrieval_module.get_stats()
                remaining_docs = stats_after.get('total_documents', 0)
                
                if remaining_docs == 0:
                    # 오류는 발생했지만 문서는 모두 삭제됨
                    logger.info(f"Documents were successfully deleted despite error: {total_documents} → 0")
                    
                    return BulkDeleteAllResponse(
                        deleted_count=total_documents,
                        collection_cleared=True,  # 결과적으로 클리어됨
                        operation_time_seconds=round(operation_time, 2),
                        message=f"All {total_documents} documents deleted successfully (collection recreated)",
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    # 부분 삭제 또는 실패
                    deleted_count = total_documents - remaining_docs
                    raise HTTPException(
                        status_code=500,
                        detail=f"Partial deletion: {deleted_count}/{total_documents} documents deleted. {remaining_docs} remaining."
                    )
                    
            except Exception as verify_error:
                # 상태 확인도 실패 - 더 심각한 문제
                logger.error(f"Cannot verify deletion status: {verify_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Deletion process failed and status verification failed. Original error: {deletion_error}"
                )
        
        # 6. 삭제 후 상태 확인 (정상 케이스)
        try:
            stats_after = await retrieval_module.get_stats()
            remaining_docs = stats_after.get('total_documents', 0)
        except Exception:
            # 통계 조회 실패 시 기본값
            remaining_docs = 0
        
        operation_time = (datetime.now() - start_time).total_seconds()
        
        # 7. 결과 검증
        if remaining_docs > 0:
            deleted_count = total_documents - remaining_docs
            logger.warning(f"Partial deletion: {deleted_count}/{total_documents} documents deleted")
            
            return BulkDeleteAllResponse(
                deleted_count=deleted_count,
                collection_cleared=False,
                operation_time_seconds=round(operation_time, 2),
                message=f"Partial deletion: {deleted_count} documents deleted, {remaining_docs} remaining",
                timestamp=datetime.now().isoformat()
            )
        
        success_message = f"Successfully deleted all {total_documents} documents"
        if total_vectors > 0:
            success_message += f" and {total_vectors} vectors"
        if request.reason:
            success_message += f" (Reason: {request.reason})"
        
        logger.info(f"BULK DELETE ALL completed: {total_documents} deleted in {operation_time:.2f}s")
        
        return BulkDeleteAllResponse(
            deleted_count=total_documents,
            collection_cleared=collection_cleared,
            operation_time_seconds=round(operation_time, 2),
            message=success_message,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Bulk delete all error: {error}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete all documents: {str(error)}"
        )

@router.post("/documents/clear-collection")
async def clear_collection_safe():
    """
    안전한 컬렉션 클리어 (관리자용)
    
    개발/테스트 환경에서만 사용 권장
    """
    try:
        # 환경 체크
        app_config = config.get('app', {})
        debug_mode = app_config.get('debug', False)
        
        if not debug_mode:
            raise HTTPException(
                status_code=403, 
                detail="Collection clearing is only allowed in debug mode"
            )
        
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        # 컬렉션 재생성 (모든 데이터 삭제 + 스키마 재생성)
        await retrieval_module.recreate_collection()
        
        logger.info("Collection cleared and recreated in debug mode")
        
        return {
            "message": "Collection cleared successfully",
            "debug_mode": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Clear collection error: {error}")
        raise HTTPException(status_code=500, detail="Failed to clear collection")

@router.post("/documents/backup-metadata")
async def backup_document_metadata():
    """
    문서 메타데이터 백업
    
    삭제 전 문서 정보를 백업하여 복구 가능하도록 함
    """
    try:
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        # 메타데이터 백업
        backup_data = await retrieval_module.backup_metadata()
        
        logger.info(f"Document metadata backed up: {len(backup_data)} documents")
        
        return {
            "message": "Metadata backup completed",
            "document_count": len(backup_data),
            "backup_size_kb": len(str(backup_data)) / 1024,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Backup metadata error: {error}")
        raise HTTPException(status_code=500, detail="Failed to backup metadata")
"""
Upload API endpoints
파일 업로드 및 문서 처리 API 엔드포인트
"""
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

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

# 업로드 상태 저장소 (실제 운영에서는 Redis 등 사용)
upload_jobs: Dict[str, Dict[str, Any]] = {}

class DocumentInfo(BaseModel):
    """문서 정보 모델"""
    id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: str
    status: str
    chunk_count: Optional[int] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class UploadResponse(BaseModel):
    """업로드 응답 모델"""
    job_id: str
    message: str
    filename: str
    file_size: int
    estimated_processing_time: float
    timestamp: str

class JobStatusResponse(BaseModel):
    """작업 상태 응답 모델"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    message: str
    filename: str
    chunk_count: Optional[int] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: str

class DocumentListResponse(BaseModel):
    """문서 목록 응답 모델"""
    documents: List[DocumentInfo]
    total_count: int
    page: int
    page_size: int
    has_next: bool

class BulkDeleteRequest(BaseModel):
    """벌크 삭제 요청 모델"""
    ids: List[str] = Field(..., description="삭제할 문서 ID 목록")

class BulkDeleteResponse(BaseModel):
    """벌크 삭제 응답 모델"""
    deleted_count: int
    failed_count: int
    failed_ids: List[str] = []
    message: str
    timestamp: str

def get_upload_directory() -> Path:
    """업로드 디렉토리 반환"""
    upload_dir = Path(__file__).parent.parent.parent.parent / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    # temp 디렉토리도 생성
    temp_dir = upload_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    return upload_dir

def estimate_processing_time(file_size: int, file_type: str) -> float:
    """파일 크기와 타입을 기반으로 처리 시간 예측"""
    base_time = 2.0  # 기본 2초
    
    # 파일 크기당 추가 시간 (MB당 1초)
    size_factor = (file_size / (1024 * 1024)) * 1.0
    
    # 파일 타입별 계수
    type_factors = {
        'pdf': 1.5,
        'docx': 1.2,
        'xlsx': 2.0,
        'txt': 0.5,
        'md': 0.5,
        'html': 0.8,
        'csv': 1.0
    }
    
    ext = file_type.lower()
    type_factor = type_factors.get(ext, 1.0)
    
    return base_time + (size_factor * type_factor)

def validate_file(file: UploadFile) -> Dict[str, Any]:
    """파일 검증"""
    # 지원하는 파일 타입
    supported_types = {
        'application/pdf': 'pdf',
        'text/plain': 'txt',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'text/csv': 'csv',
        'text/html': 'html',
        'text/markdown': 'md'
    }
    
    # MIME 타입 검증
    if file.content_type not in supported_types:
        # 확장자로 재시도
        ext = Path(file.filename).suffix.lower()[1:]  # 점 제거
        if ext not in supported_types.values():
            return {
                "valid": False,
                "error": f"Unsupported file type: {file.content_type}. Supported types: {', '.join(supported_types.keys())}"
            }
        file_type = ext
    else:
        file_type = supported_types[file.content_type]
    
    # 파일 크기 제한 (기본 10MB)
    max_size = config.get('upload', {}).get('max_file_size', 10 * 1024 * 1024)
    if file.size and file.size > max_size:
        return {
            "valid": False,
            "error": f"File too large: {file.size} bytes. Maximum allowed: {max_size} bytes"
        }
    
    return {
        "valid": True,
        "file_type": file_type
    }

async def process_document_background(job_id: str, file_path: Path, filename: str, file_type: str):
    """백그라운드 문서 처리"""
    try:
        # 작업 상태 업데이트
        upload_jobs[job_id].update({
            "status": "processing",
            "progress": 10,
            "message": "문서 처리 시작..."
        })
        
        document_processor = modules.get('document_processor')
        retrieval_module = modules.get('retrieval')
        
        if not document_processor or not retrieval_module:
            raise Exception("Required modules not available")
        
        # 문서 로드
        logger.info(f"Loading document: {filename}")
        upload_jobs[job_id].update({
            "progress": 30,
            "message": "문서 로딩 중..."
        })
        
        docs = await document_processor.load_document(str(file_path), {
            "source_file": filename,
            "file_type": file_type
        })
        
        # 문서 분할
        logger.info(f"Splitting document into chunks: {len(docs)} documents")
        upload_jobs[job_id].update({
            "progress": 50,
            "message": "문서 분할 중..."
        })
        
        chunks = await document_processor.split_documents(docs)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # 임베딩 생성
        upload_jobs[job_id].update({
            "progress": 70,
            "message": "임베딩 생성 중..."
        })
        
        embedded_chunks = await document_processor.embed_chunks(chunks)
        
        # 벡터 스토어에 저장
        upload_jobs[job_id].update({
            "progress": 90,
            "message": "벡터 DB에 저장 중..."
        })
        
        await retrieval_module.add_documents(embedded_chunks)
        
        # 임시 파일 삭제
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")
        
        # 완료 상태 업데이트
        processing_time = datetime.now().timestamp() - upload_jobs[job_id]["start_time"]
        upload_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "message": "문서 처리 완료",
            "chunk_count": len(chunks),
            "processing_time": processing_time
        })
        
        logger.info(f"Document processing completed: {filename}, {len(chunks)} chunks, {processing_time:.2f}s")
        
    except Exception as error:
        logger.error(f"Document processing failed: {error}")
        
        # 에러 상태 업데이트
        upload_jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "message": "문서 처리 실패",
            "error_message": str(error)
        })
        
        # 임시 파일 삭제 시도
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception:
            pass

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """문서 업로드"""
    try:
        # 파일 검증
        validation = validate_file(file)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])
        
        file_type = validation["file_type"]
        
        # 작업 ID 생성
        job_id = str(uuid4())
        
        # 임시 파일 저장
        upload_dir = get_upload_directory()
        temp_dir = upload_dir / "temp"
        file_path = temp_dir / f"{job_id}_{file.filename}"
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = len(content)
        
        # 작업 상태 초기화
        upload_jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_size": file_size,
            "status": "pending",
            "progress": 0,
            "message": "업로드 완료, 처리 대기 중...",
            "start_time": datetime.now().timestamp(),
            "chunk_count": None,
            "processing_time": None,
            "error_message": None
        }
        
        # 백그라운드 처리 시작
        background_tasks.add_task(
            process_document_background,
            job_id,
            file_path,
            file.filename,
            file_type
        )
        
        # 처리 시간 예측
        estimated_time = estimate_processing_time(file_size, file_type)
        
        logger.info(f"Document upload initiated: {file.filename}, job_id: {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            message="File uploaded successfully and processing started",
            filename=file.filename,
            file_size=file_size,
            estimated_processing_time=estimated_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as error:
        logger.error(f"Upload error: {error}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(error)}")

@router.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def get_upload_status(job_id: str):
    """업로드 작업 상태 조회"""
    if job_id not in upload_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = upload_jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        filename=job["filename"],
        chunk_count=job["chunk_count"],
        processing_time=job["processing_time"],
        error_message=job["error_message"],
        timestamp=datetime.now().isoformat()
    )

@router.get("/upload/documents", response_model=DocumentListResponse)
async def list_documents(page: int = 1, page_size: int = 20):
    """문서 목록 조회"""
    try:
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        logger.info(f"Listing documents: page={page}, page_size={page_size}")
        
        # 벡터 스토어에서 문서 목록 조회
        documents_data = await retrieval_module.list_documents(page=page, page_size=page_size)
        logger.info(f"Retrieved documents_data: {documents_data}")
        
        documents = []
        for doc_data in documents_data.get("documents", []):
            # upload_date 처리: timestamp가 float인 경우 ISO 형식으로 변환
            upload_date = doc_data.get("upload_date", datetime.now().timestamp())
            if isinstance(upload_date, (int, float)):
                upload_date = datetime.fromtimestamp(upload_date).isoformat()
            elif not upload_date:
                upload_date = datetime.now().isoformat()
                
            documents.append(DocumentInfo(
                id=doc_data.get("id", "unknown"),
                filename=doc_data.get("filename", "unknown"),
                file_type=doc_data.get("file_type", "unknown"),
                file_size=doc_data.get("file_size", 0),
                upload_date=upload_date,
                status="completed",
                chunk_count=doc_data.get("chunk_count", 0)
            ))
        
        total_count = documents_data.get("total_count", len(documents))
        
        response = DocumentListResponse(
            documents=documents,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total_count
        )
        
        logger.info(f"Returning response: {len(documents)} documents, total={total_count}")
        return response
        
    except Exception as error:
        logger.error(f"List documents error: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(error)}")

@router.delete("/upload/documents/{document_id}")
async def delete_document(document_id: str):
    """문서 삭제"""
    try:
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        await retrieval_module.delete_document(document_id)
        
        logger.info(f"Document deleted: {document_id}")
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Delete document error: {error}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/upload/documents/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_documents(request: BulkDeleteRequest):
    """문서 일괄 삭제"""
    try:
        retrieval_module = modules.get('retrieval')
        if not retrieval_module:
            raise HTTPException(status_code=500, detail="Retrieval module not available")
        
        deleted_count = 0
        failed_count = 0
        failed_ids = []
        
        logger.info(f"Bulk delete requested for {len(request.ids)} documents: {request.ids}")
        
        for document_id in request.ids:
            try:
                # null 또는 빈 문자열 체크
                if not document_id or document_id.strip() == "":
                    logger.warning(f"Skipping invalid document ID: {document_id}")
                    failed_count += 1
                    failed_ids.append(document_id)
                    continue
                    
                await retrieval_module.delete_document(document_id)
                deleted_count += 1
                logger.info(f"Successfully deleted document: {document_id}")
                
            except Exception as delete_error:
                logger.error(f"Failed to delete document {document_id}: {delete_error}")
                failed_count += 1
                failed_ids.append(document_id)
        
        message = f"Bulk delete completed: {deleted_count} deleted, {failed_count} failed"
        logger.info(message)
        
        return BulkDeleteResponse(
            deleted_count=deleted_count,
            failed_count=failed_count,
            failed_ids=failed_ids,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as error:
        logger.error(f"Bulk delete error: {error}")
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {str(error)}")

@router.get("/upload/supported-types")
async def get_supported_types():
    """지원하는 파일 타입 목록"""
    return {
        "supported_types": {
            "pdf": {
                "mime_type": "application/pdf",
                "description": "PDF documents",
                "max_size_mb": 10
            },
            "docx": {
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "description": "Microsoft Word documents",
                "max_size_mb": 10
            },
            "xlsx": {
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                "description": "Microsoft Excel spreadsheets",
                "max_size_mb": 10
            },
            "txt": {
                "mime_type": "text/plain",
                "description": "Plain text files",
                "max_size_mb": 10
            },
            "csv": {
                "mime_type": "text/csv",
                "description": "Comma-separated values",
                "max_size_mb": 10
            },
            "html": {
                "mime_type": "text/html",
                "description": "HTML documents",
                "max_size_mb": 10
            },
            "md": {
                "mime_type": "text/markdown",
                "description": "Markdown documents",
                "max_size_mb": 10
            }
        },
        "max_file_size": config.get('upload', {}).get('max_file_size', 10 * 1024 * 1024),
        "max_files_per_request": 1
    }
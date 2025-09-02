"""
Simple API Tests (without complex dependencies)
간단한 API 테스트 (복잡한 의존성 없이)
"""
import pytest
from fastapi.testclient import TestClient


class TestSimpleAPIEndpoints:
    """간단한 API 엔드포인트 테스트"""

    def test_health_check_basic(self, simple_test_client: TestClient):
        """기본 헬스 체크 테스트"""
        response = simple_test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_check_response_format(self, simple_test_client: TestClient):
        """헬스 체크 응답 형식 테스트"""
        response = simple_test_client.get("/health")
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) >= 3  # status, timestamp, version 최소

    def test_stats_endpoint_basic(self, simple_test_client: TestClient):
        """기본 통계 엔드포인트 테스트"""
        response = simple_test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_fields = [
            "total_documents",
            "total_chunks", 
            "total_queries",
            "api_usage",
            "system_stats"
        ]
        
        for field in expected_fields:
            assert field in data

    def test_upload_endpoint_basic(self, simple_test_client: TestClient):
        """기본 업로드 엔드포인트 테스트"""
        response = simple_test_client.post("/api/upload")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert data["success"] is True
        assert "job_id" in data
        assert "message" in data

    def test_documents_list_basic(self, simple_test_client: TestClient):
        """기본 문서 목록 테스트"""
        response = simple_test_client.get("/api/upload/documents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
        assert isinstance(data["total"], int)

    def test_chat_endpoint_basic(self, simple_test_client: TestClient):
        """기본 채팅 엔드포인트 테스트"""
        response = simple_test_client.post("/api/chat")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "session_id" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_new_session_basic(self, simple_test_client: TestClient):
        """기본 새 세션 테스트"""
        response = simple_test_client.post("/api/chat/session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessionId" in data
        session_id = data["sessionId"]
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_api_response_consistency(self, simple_test_client: TestClient):
        """API 응답 일관성 테스트"""
        endpoints = [
            ("/health", "GET"),
            ("/stats", "GET"),
            ("/api/upload/documents", "GET"),
        ]
        
        for endpoint, method in endpoints:
            if method == "GET":
                response = simple_test_client.get(endpoint)
            else:
                response = simple_test_client.post(endpoint)
            
            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")
            
            # JSON 파싱이 가능해야 함
            data = response.json()
            assert isinstance(data, dict)

    def test_api_error_handling(self, simple_test_client: TestClient):
        """API 에러 처리 테스트"""
        # 존재하지 않는 엔드포인트
        response = simple_test_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # 잘못된 HTTP 메서드
        response = simple_test_client.delete("/health")
        assert response.status_code in [405, 404]  # Method Not Allowed or Not Found

    def test_concurrent_requests(self, simple_test_client: TestClient):
        """동시 요청 테스트"""
        import concurrent.futures
        
        def make_health_request():
            return simple_test_client.get("/health")
        
        # 5개의 동시 요청
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_health_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # 모든 요청이 성공해야 함
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_api_performance(self, simple_test_client: TestClient, performance_timer):
        """API 성능 테스트"""
        performance_timer.start()
        response = simple_test_client.get("/health")
        performance_timer.stop()
        
        assert response.status_code == 200
        # 헬스 체크는 100ms 이내에 완료되어야 함
        assert performance_timer.elapsed < 0.1

    @pytest.mark.parametrize("endpoint", [
        "/health",
        "/stats", 
        "/api/upload/documents"
    ])
    def test_endpoint_availability(self, simple_test_client: TestClient, endpoint):
        """엔드포인트 가용성 테스트"""
        response = simple_test_client.get(endpoint)
        assert response.status_code == 200
        assert response.json() is not None

    def test_json_response_format(self, simple_test_client: TestClient):
        """JSON 응답 형식 테스트"""
        response = simple_test_client.get("/health")
        
        # Content-Type 헤더 확인
        content_type = response.headers.get("content-type")
        assert "application/json" in content_type
        
        # JSON 파싱 확인
        data = response.json()
        assert isinstance(data, dict)
        
        # 기본 필드 타입 확인
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_api_consistency_across_methods(self, simple_test_client: TestClient):
        """HTTP 메서드 간 일관성 테스트"""
        # GET 요청들
        get_endpoints = ["/health", "/stats", "/api/upload/documents"]
        
        for endpoint in get_endpoints:
            response = simple_test_client.get(endpoint)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
        
        # POST 요청들
        post_endpoints = ["/api/upload", "/api/chat", "/api/chat/session"]
        
        for endpoint in post_endpoints:
            response = simple_test_client.post(endpoint)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
"""
Chat API Tests
채팅 API 테스트
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


class TestChatAPI:
    """채팅 API 테스트 클래스"""

    def test_send_message_success(self, test_client: TestClient, sample_chat_message):
        """메시지 전송 성공 테스트"""
        response = test_client.post("/api/chat", json=sample_chat_message)
        
        assert response.status_code == 200
        data = response.json()
        
        # 응답 필드 검증
        assert "response" in data
        assert "session_id" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_send_message_without_session(self, test_client: TestClient):
        """세션 ID 없이 메시지 전송 테스트"""
        message_data = {"message": "안녕하세요!"}
        response = test_client.post("/api/chat", json=message_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # 새 세션 ID가 생성되어야 함
        assert "session_id" in data
        assert "response" in data

    def test_send_empty_message(self, test_client: TestClient):
        """빈 메시지 전송 테스트"""
        message_data = {"message": "", "session_id": "test-session"}
        response = test_client.post("/api/chat", json=message_data)
        
        assert response.status_code == 400

    def test_send_long_message(self, test_client: TestClient):
        """긴 메시지 전송 테스트"""
        long_message = "테스트 " * 1000  # 매우 긴 메시지
        message_data = {"message": long_message, "session_id": "test-session"}
        
        response = test_client.post("/api/chat", json=message_data)
        
        # 메시지 길이 제한에 따라 처리
        assert response.status_code in [200, 400]

    def test_send_message_with_special_characters(self, test_client: TestClient):
        """특수 문자가 포함된 메시지 전송 테스트"""
        special_message = "안녕하세요! 🤖 @#$%^&*()_+ 한국어 테스트입니다."
        message_data = {"message": special_message, "session_id": "test-session"}
        
        response = test_client.post("/api/chat", json=message_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_get_chat_history(self, test_client: TestClient):
        """채팅 기록 조회 테스트"""
        session_id = "test-session-123"
        response = test_client.get(f"/api/chat/history/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_get_chat_history_invalid_session(self, test_client: TestClient):
        """유효하지 않은 세션으로 기록 조회 테스트"""
        invalid_session_id = "invalid-session-id"
        response = test_client.get(f"/api/chat/history/{invalid_session_id}")
        
        assert response.status_code in [200, 404]

    def test_start_new_session(self, test_client: TestClient):
        """새 채팅 세션 시작 테스트"""
        response = test_client.post("/api/chat/session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessionId" in data or "session_id" in data
        session_id = data.get("sessionId") or data.get("session_id")
        assert session_id is not None
        assert len(session_id) > 0

    def test_chat_with_context(self, test_client: TestClient):
        """문맥이 있는 대화 테스트"""
        session_id = "context-test-session"
        
        # 첫 번째 메시지
        message1 = {"message": "RAG 시스템에 대해 설명해주세요.", "session_id": session_id}
        response1 = test_client.post("/api/chat", json=message1)
        assert response1.status_code == 200
        
        # 두 번째 메시지 (이전 대화 참조)
        message2 = {"message": "더 자세히 설명해주세요.", "session_id": session_id}
        response2 = test_client.post("/api/chat", json=message2)
        assert response2.status_code == 200

    def test_chat_response_time(self, test_client: TestClient, performance_timer):
        """채팅 응답 시간 테스트"""
        message_data = {"message": "간단한 질문입니다.", "session_id": "perf-test"}
        
        performance_timer.start()
        response = test_client.post("/api/chat", json=message_data)
        performance_timer.stop()
        
        assert response.status_code == 200
        # 응답 시간 30초 이내 (Mock이므로 매우 빠를 것)
        assert performance_timer.elapsed < 30.0

    def test_chat_with_source_references(self, test_client: TestClient):
        """소스 참조가 있는 채팅 응답 테스트"""
        message_data = {
            "message": "업로드된 문서에서 정보를 찾아주세요.",
            "session_id": "source-test"
        }
        
        response = test_client.post("/api/chat", json=message_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if "sources" in data and data["sources"]:
            # 소스가 있는 경우 구조 검증
            for source in data["sources"]:
                assert "text" in source or "content" in source
                assert "score" in source

    def test_malformed_chat_request(self, test_client: TestClient):
        """잘못된 형식의 채팅 요청 테스트"""
        # 필수 필드 누락
        invalid_data = {"invalid_field": "value"}
        response = test_client.post("/api/chat", json=invalid_data)
        
        assert response.status_code == 422

    def test_chat_rate_limiting(self, test_client: TestClient):
        """채팅 요청 속도 제한 테스트"""
        message_data = {"message": "속도 제한 테스트", "session_id": "rate-limit-test"}
        
        # 연속으로 많은 요청 전송
        responses = []
        for _ in range(10):
            response = test_client.post("/api/chat", json=message_data)
            responses.append(response.status_code)
        
        # 일부 요청은 성공해야 하고, 속도 제한에 걸릴 수 있음
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        assert success_count > 0  # 최소 일부 요청은 성공
        # 속도 제한이 적용되면 429 상태 코드 반환

    @pytest.mark.slow
    def test_concurrent_chat_requests(self, test_client: TestClient):
        """동시 채팅 요청 테스트"""
        import concurrent.futures
        
        def send_message(session_id):
            message_data = {
                "message": f"동시 요청 테스트 - 세션 {session_id}",
                "session_id": f"concurrent-{session_id}"
            }
            return test_client.post("/api/chat", json=message_data)
        
        # 5개의 동시 요청
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_message, i) for i in range(5)]
            responses = [future.result() for future in futures]
        
        # 모든 요청이 처리되어야 함
        for response in responses:
            assert response.status_code in [200, 429]  # 성공 또는 속도 제한

    def test_chat_session_persistence(self, test_client: TestClient):
        """채팅 세션 지속성 테스트"""
        # 새 세션 생성
        session_response = test_client.post("/api/chat/session")
        assert session_response.status_code == 200
        
        session_data = session_response.json()
        session_id = session_data.get("sessionId") or session_data.get("session_id")
        
        # 메시지 전송
        message_data = {"message": "세션 테스트", "session_id": session_id}
        chat_response = test_client.post("/api/chat", json=message_data)
        assert chat_response.status_code == 200
        
        # 기록 조회
        history_response = test_client.get(f"/api/chat/history/{session_id}")
        assert history_response.status_code == 200

    def test_chat_websocket_upgrade(self, test_client: TestClient):
        """WebSocket 업그레이드 테스트"""
        # WebSocket 연결 시도
        with test_client.websocket_connect("/api/chat/ws") as websocket:
            # 연결 성공하면 메시지 전송 테스트
            test_message = {
                "type": "message",
                "content": "WebSocket 테스트",
                "session_id": "ws-test"
            }
            websocket.send_json(test_message)
            
            # 응답 받기
            response = websocket.receive_json()
            assert "type" in response
            assert "content" in response

    def test_chat_error_handling(self, test_client: TestClient):
        """채팅 에러 처리 테스트"""
        # 매우 긴 메시지로 에러 유발
        very_long_message = "A" * 100000  # 100KB 메시지
        message_data = {
            "message": very_long_message,
            "session_id": "error-test"
        }
        
        response = test_client.post("/api/chat", json=message_data)
        
        # 에러 응답 확인
        assert response.status_code in [400, 413, 422]

    def test_chat_multilingual_support(self, test_client: TestClient):
        """다국어 지원 테스트"""
        messages = [
            "안녕하세요, 한국어 테스트입니다.",
            "Hello, this is an English test.",
            "こんにちは、日本語のテストです。",
            "你好，这是中文测试。"
        ]
        
        for i, message in enumerate(messages):
            message_data = {
                "message": message,
                "session_id": f"multilingual-{i}"
            }
            response = test_client.post("/api/chat", json=message_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
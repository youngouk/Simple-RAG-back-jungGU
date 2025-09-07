#!/usr/bin/env python3
"""
대화 기록 메모리 테스트 스크립트
LangChain ConversationBufferMemory 통합 테스트
"""
import asyncio
import httpx
import json
from datetime import datetime

# API 엔드포인트
BASE_URL = "http://localhost:8000"

async def test_conversation_memory():
    """대화 기록 메모리 테스트"""
    async with httpx.AsyncClient() as client:
        session_id = None
        
        print("=" * 60)
        print("🧪 대화 기록 메모리 테스트 시작")
        print("=" * 60)
        
        # 1. 첫 번째 메시지: 이름 소개
        print("\n1️⃣ 첫 번째 메시지: 이름 소개")
        response = await client.post(
            f"{BASE_URL}/api/chat",
            json={"message": "안녕! 내 이름은 김철수야."}
        )
        result = response.json()
        session_id = result.get("session_id")
        print(f"   세션 ID: {session_id}")
        print(f"   응답: {result.get('answer', '')[:100]}...")
        
        await asyncio.sleep(2)  # 잠시 대기
        
        # 2. 두 번째 메시지: 이름 확인
        print("\n2️⃣ 두 번째 메시지: 이름 확인 (이전 대화 기억 테스트)")
        response = await client.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "내 이름이 뭐라고 했지?",
                "session_id": session_id
            }
        )
        result = response.json()
        answer = result.get("answer", "")
        print(f"   응답: {answer[:200]}...")
        
        # 이름 기억 확인
        if "김철수" in answer or "철수" in answer:
            print("   ✅ 성공: 이름을 기억하고 있습니다!")
        else:
            print("   ❌ 실패: 이름을 기억하지 못합니다.")
        
        await asyncio.sleep(2)
        
        # 3. 세 번째 메시지: 추가 정보 제공
        print("\n3️⃣ 세 번째 메시지: 추가 정보 제공")
        response = await client.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "나는 서울에 살고 있고, 개발자야.",
                "session_id": session_id
            }
        )
        result = response.json()
        print(f"   응답: {result.get('answer', '')[:100]}...")
        
        await asyncio.sleep(2)
        
        # 4. 네 번째 메시지: 전체 정보 확인
        print("\n4️⃣ 네 번째 메시지: 전체 정보 확인")
        response = await client.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "내가 지금까지 너에게 알려준 정보를 요약해줄래?",
                "session_id": session_id
            }
        )
        result = response.json()
        answer = result.get("answer", "")
        print(f"   응답: {answer[:300]}...")
        
        # 정보 기억 확인
        remembered_info = []
        if "김철수" in answer or "철수" in answer:
            remembered_info.append("이름(김철수)")
        if "서울" in answer:
            remembered_info.append("거주지(서울)")
        if "개발자" in answer:
            remembered_info.append("직업(개발자)")
        
        print(f"\n   📊 기억된 정보: {', '.join(remembered_info) if remembered_info else '없음'}")
        
        # 5. 세션 통계 확인
        print("\n5️⃣ 세션 통계 확인")
        response = await client.get(f"{BASE_URL}/api/stats/session")
        if response.status_code == 200:
            stats = response.json()
            print(f"   활성 세션 수: {stats.get('active_sessions', 0)}")
            print(f"   총 대화 수: {stats.get('total_conversations', 0)}")
            print(f"   LangChain 메모리 사용: {stats.get('use_langchain_memory', False)}")
            print(f"   메모리 타입: {stats.get('memory_type', 'N/A')}")
            print(f"   최대 대화 기억 수: {stats.get('max_conversation_memory', 0)}")
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("📋 테스트 결과 요약")
        print("=" * 60)
        if len(remembered_info) >= 2:
            print("✅ 대화 기록 메모리가 정상적으로 작동합니다!")
            print(f"   - 기억된 정보: {', '.join(remembered_info)}")
        else:
            print("⚠️ 대화 기록 메모리가 부분적으로만 작동합니다.")
            print(f"   - 기억된 정보: {', '.join(remembered_info) if remembered_info else '없음'}")

if __name__ == "__main__":
    asyncio.run(test_conversation_memory())
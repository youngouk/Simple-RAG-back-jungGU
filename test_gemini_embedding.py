"""
Gemini Embedding 001 테스트 스크립트
1536차원 임베딩 생성 및 L2 정규화 검증
"""
import asyncio
import numpy as np
import os
from app.modules.gemini_embeddings import GeminiEmbeddings

async def test_gemini_embeddings():
    """Gemini Embedding 테스트"""
    
    # API 키 설정 (환경 변수에서 읽기)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        return False
    
    print("🚀 Gemini Embedding 001 테스트 시작...")
    print("-" * 50)
    
    try:
        # GeminiEmbeddings 인스턴스 생성
        embedder = GeminiEmbeddings(
            google_api_key=api_key,
            model_name="models/gemini-embedding-001",
            output_dimensionality=1536,
            batch_size=100
        )
        print("✅ GeminiEmbeddings 인스턴스 생성 완료")
        
        # 테스트 텍스트
        test_query = "한국의 아름다운 계절은 가을입니다."
        test_documents = [
            "가을은 단풍이 아름다운 계절입니다.",
            "봄에는 꽃이 피고 날씨가 따뜻합니다.",
            "여름은 덥고 습한 날씨가 특징입니다."
        ]
        
        print("\n📝 테스트 데이터:")
        print(f"  Query: {test_query}")
        print(f"  Documents: {len(test_documents)}개")
        
        # 1. 쿼리 임베딩 테스트
        print("\n🔍 쿼리 임베딩 생성 중...")
        query_embedding = embedder.embed_query(test_query)
        
        print(f"  ✅ 차원: {len(query_embedding)}")
        assert len(query_embedding) == 1536, f"차원 오류: {len(query_embedding)} != 1536"
        
        # L2 norm 확인
        query_norm = np.linalg.norm(np.array(query_embedding))
        print(f"  ✅ L2 Norm: {query_norm:.6f}")
        assert abs(query_norm - 1.0) < 0.01, f"정규화 오류: norm={query_norm}"
        
        # 2. 문서 임베딩 테스트
        print("\n📚 문서 임베딩 생성 중...")
        doc_embeddings = embedder.embed_documents(test_documents)
        
        print(f"  ✅ 생성된 임베딩 수: {len(doc_embeddings)}")
        assert len(doc_embeddings) == len(test_documents), "문서 수 불일치"
        
        for i, embedding in enumerate(doc_embeddings):
            doc_norm = np.linalg.norm(np.array(embedding))
            print(f"  문서 {i+1} - 차원: {len(embedding)}, L2 Norm: {doc_norm:.6f}")
            assert len(embedding) == 1536, f"문서 {i+1} 차원 오류"
            assert abs(doc_norm - 1.0) < 0.01, f"문서 {i+1} 정규화 오류"
        
        # 3. 비동기 메서드 테스트
        print("\n⚡ 비동기 메서드 테스트...")
        async_query_embedding = await embedder.aembed_query(test_query)
        async_doc_embeddings = await embedder.aembed_documents(test_documents[:1])
        
        print(f"  ✅ 비동기 쿼리 임베딩 차원: {len(async_query_embedding)}")
        print(f"  ✅ 비동기 문서 임베딩 수: {len(async_doc_embeddings)}")
        
        # 4. 유사도 계산 테스트
        print("\n📊 코사인 유사도 계산:")
        query_vec = np.array(query_embedding)
        
        for i, doc_embedding in enumerate(doc_embeddings):
            doc_vec = np.array(doc_embedding)
            # 코사인 유사도 계산 (정규화된 벡터이므로 내적이 코사인 유사도)
            similarity = np.dot(query_vec, doc_vec)
            print(f"  문서 {i+1}: {test_documents[i][:30]}... → 유사도: {similarity:.4f}")
        
        # 5. 임베딩 검증
        print("\n🔎 임베딩 검증:")
        for i, embedding in enumerate([query_embedding] + doc_embeddings):
            is_valid = embedder.validate_embedding(embedding)
            label = "Query" if i == 0 else f"Doc{i}"
            status = "✅" if is_valid else "❌"
            print(f"  {label}: {status} Valid")
        
        print("\n" + "=" * 50)
        print("✨ 모든 테스트 통과!")
        print(f"📏 임베딩 차원: 1536")
        print(f"🎯 L2 정규화: 완료")
        print(f"🔄 Task Types: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 테스트 실행
    success = asyncio.run(test_gemini_embeddings())
    exit(0 if success else 1)
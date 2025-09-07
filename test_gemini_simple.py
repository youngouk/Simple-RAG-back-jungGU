"""
간단한 Gemini Embedding 테스트
"""
import os
import numpy as np
from app.modules.gemini_embeddings import GeminiEmbeddings

def test_simple():
    api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCdbgcPUgo9_ScSB1N3_VH4vWi2ofiz2hY")
    
    embedder = GeminiEmbeddings(
        google_api_key=api_key,
        model_name="models/gemini-embedding-001",
        output_dimensionality=1536,
        batch_size=100
    )
    
    # 테스트 텍스트
    test_documents = [
        "첫 번째 문서입니다.",
        "두 번째 문서입니다.",
        "세 번째 문서입니다."
    ]
    
    print("📚 문서 임베딩 생성 중...")
    doc_embeddings = embedder.embed_documents(test_documents)
    
    print(f"반환된 타입: {type(doc_embeddings)}")
    print(f"리스트 길이: {len(doc_embeddings)}")
    
    for i, embedding in enumerate(doc_embeddings):
        print(f"\n문서 {i+1}:")
        print(f"  타입: {type(embedding)}")
        print(f"  차원: {len(embedding)}")
        if isinstance(embedding, list) and len(embedding) > 0:
            print(f"  첫 번째 요소 타입: {type(embedding[0])}")
            if isinstance(embedding[0], (int, float)):
                print(f"  벡터 샘플: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
                norm = np.linalg.norm(np.array(embedding))
                print(f"  L2 Norm: {norm:.6f}")

if __name__ == "__main__":
    test_simple()
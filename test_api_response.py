"""
Gemini API 응답 형식 테스트
"""
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCdbgcPUgo9_ScSB1N3_VH4vWi2ofiz2hY")
genai.configure(api_key=api_key)

# 테스트 텍스트
texts = ["첫 번째", "두 번째", "세 번째"]

# API 호출
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=texts,
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=1536
)

print(f"Result type: {type(result)}")
print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'No keys'}")

if 'embedding' in result:
    print(f"'embedding' type: {type(result['embedding'])}")
    print(f"'embedding' length: {len(result['embedding'])}")
    if isinstance(result['embedding'], list) and len(result['embedding']) > 0:
        print(f"First element type: {type(result['embedding'][0])}")
        if isinstance(result['embedding'][0], list):
            print(f"First element length: {len(result['embedding'][0])}")

if 'embeddings' in result:
    print(f"'embeddings' type: {type(result['embeddings'])}")
    print(f"'embeddings' length: {len(result['embeddings'])}")
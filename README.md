# RAG Chatbot Backend

Python FastAPI 기반의 한국어 RAG 챗봇 백엔드 서버입니다.

## 특징

- FastAPI 기반 REST API 서버
- Multi-LLM 지원 (Google Gemini, Anthropic Claude, OpenAI)
- Qdrant 벡터 데이터베이스 연동
- 하이브리드 검색 (Dense + Sparse)
- 비동기 처리 및 세션 관리

## 실행 방법

```bash
# 개발 서버 실행
make dev

# 프로덕션 서버 실행
make run
```

## 환경 변수

- `QDRANT_URL`: Qdrant 서버 URL
- `QDRANT_API_KEY`: Qdrant API 키
- `GOOGLE_API_KEY`: Google Gemini API 키
- `ANTHROPIC_API_KEY`: Anthropic Claude API 키
- `OPENAI_API_KEY`: OpenAI API 키
.PHONY: help install install-dev sync update run dev test lint format clean docker-build docker-run

# 기본 타겟
.DEFAULT_GOAL := help

# 도움말
help:
	@echo "RAG Chatbot Python Backend - Makefile Commands"
	@echo "=============================================="
	@echo "install      - uv로 프로덕션 의존성 설치"
	@echo "install-dev  - uv로 개발 의존성 포함 설치"
	@echo "sync         - uv.lock 파일과 동기화"
	@echo "update       - 의존성 업데이트"
	@echo "run          - 프로덕션 서버 실행"
	@echo "dev          - 개발 서버 실행 (자동 리로드)"
	@echo "test         - 테스트 실행"
	@echo "lint         - 코드 린팅 (ruff)"
	@echo "format       - 코드 포맷팅 (black + ruff)"
	@echo "clean        - 캐시 및 임시 파일 정리"
	@echo "docker-build - Docker 이미지 빌드"
	@echo "docker-run   - Docker 컨테이너 실행"

# uv 설치 확인
check-uv:
	@command -v uv >/dev/null 2>&1 || { echo "uv가 설치되어 있지 않습니다. 'curl -LsSf https://astral.sh/uv/install.sh | sh'로 설치하세요."; exit 1; }

# 프로덕션 의존성 설치
install: check-uv
	uv sync --no-dev

# 개발 의존성 포함 설치
install-dev: check-uv
	uv sync

# lock 파일과 동기화
sync: check-uv
	uv sync

# 의존성 업데이트
update: check-uv
	uv lock --upgrade

# 프로덕션 서버 실행
run: install
	uv run python main.py

# 개발 서버 실행
dev: install-dev
	uv run python main.py

# 개발 서버 실행 (uvicorn 직접 실행)
dev-reload: install-dev
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 테스트 실행 (UV 환경)
test: install-dev
	uv run pytest

# 테스트 실행 (시스템 Python 환경 - UV 문제 시 대안)
test-system:
	@echo "🐍 Using system Python environment..."
	@if [ ! -d ".venv_system" ]; then \
		echo "Creating system Python environment..."; \
		python3 -m venv .venv_system; \
		.venv_system/bin/pip install pytest pytest-asyncio fastapi pyyaml structlog psutil; \
	fi
	@source .venv_system/bin/activate && python -m pytest

# 테스트 환경 자동 설정 및 실행
test-auto:
	@echo "🚀 Auto-configuring test environment..."
	@./scripts/test-env-setup.sh --minimal
	@if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then \
		echo "Using UV environment..."; \
		source .venv/bin/activate && python -m pytest; \
	elif [ -d ".venv_system" ]; then \
		echo "Using system Python environment..."; \
		source .venv_system/bin/activate && python -m pytest; \
	else \
		echo "❌ No suitable environment found"; \
		exit 1; \
	fi

# 기본 테스트 (타임아웃 방지)
test-basic:
	@echo "🧪 Running basic tests only..."
	@if [ -d ".venv_system" ]; then \
		source .venv_system/bin/activate && python -m pytest tests/test_basic.py tests/test_config_simple.py -v --tb=short; \
	else \
		./scripts/test-env-setup.sh --system --minimal && \
		source .venv_system/bin/activate && python -m pytest tests/test_basic.py tests/test_config_simple.py -v --tb=short; \
	fi

# 테스트 커버리지
test-cov: install-dev
	uv run pytest --cov=app --cov-report=html --cov-report=term

# 테스트 환경 진단
test-env-check:
	@echo "🔍 Testing environment diagnosis..."
	@./scripts/test-env-setup.sh --verbose

# 린팅
lint: install-dev
	uv run ruff check .

# 린팅 수정
lint-fix: install-dev
	uv run ruff check --fix .

# 코드 포맷팅
format: install-dev
	uv run black .
	uv run ruff check --fix .

# 타입 체크
type-check: install-dev
	uv run mypy .

# 정리
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf uploads/temp/*

# Docker 빌드
docker-build:
	docker build -t rag-chatbot:latest .

# Docker 실행
docker-run:
	docker run -p 8000:8000 --env-file ../.env rag-chatbot:latest

# 환경 정보 출력
info: check-uv
	@echo "Python version:"
	@uv run python --version
	@echo "\nInstalled packages:"
	@uv pip list

# 개발 환경 초기 설정
setup: check-uv
	uv venv
	uv sync
	@echo "\n✅ 개발 환경 설정 완료!"
	@echo "다음 명령어로 개발 서버를 실행하세요: make dev"
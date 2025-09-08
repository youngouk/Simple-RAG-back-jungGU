# 개발 가이드 - UV를 사용한 Python 프로젝트

## 🚀 UV 소개

[UV](https://github.com/astral-sh/uv)는 Rust로 작성된 초고속 Python 패키지 및 프로젝트 관리자입니다. pip와 pip-tools를 대체하며 10-100배 빠른 성능을 제공합니다.

## 📦 프로젝트 구조

```
backend_py/
├── pyproject.toml      # 프로젝트 설정 및 의존성
├── .python-version     # Python 버전 고정
├── uv.lock            # 정확한 의존성 잠금 (자동 생성)
├── .gitignore         # Git 무시 파일
├── Makefile           # 개발 명령어 모음
├── Dockerfile         # UV 기반 Docker 이미지
└── app/               # 애플리케이션 소스
```

## 🛠️ 개발 환경 설정

### 1. UV 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip로
pip install uv
```

### 2. 프로젝트 초기화

```bash
# 전체 개발 환경 설정
make setup

# 또는 수동으로
uv venv                 # 가상환경 생성
uv sync                 # 의존성 설치
```

## 📋 주요 명령어

### 기본 작업

```bash
# 의존성 관리
uv add fastapi          # 새 패키지 추가
uv add --dev pytest     # 개발 의존성 추가
uv remove requests      # 패키지 제거
uv sync                 # lock 파일과 동기화
uv lock --upgrade       # 의존성 업데이트

# 실행
uv run python main.py   # 스크립트 실행
uv run pytest          # 테스트 실행
uv run black .          # 포맷팅
```

### Makefile 명령어

```bash
make help              # 모든 명령어 보기
make dev               # 개발 서버 시작
make test              # 테스트 실행
make format            # 코드 포맷팅
make lint              # 린팅
make clean             # 캐시 정리
```

## 🔧 의존성 관리

### pyproject.toml 구조

```toml
[project]
name = "rag-chatbot"
dependencies = [
    "fastapi>=0.104.1",    # 프로덕션 의존성
    "uvicorn[standard]>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",      # 개발 의존성
    "black>=23.12.0",
]
```

### 의존성 그룹

- **기본**: 프로덕션에 필요한 핵심 패키지
- **dev**: 개발 도구 (테스트, 린팅, 포맷팅)
- **monitoring**: 모니터링 도구 (선택적)
- **security**: 보안 관련 패키지 (선택적)
- **redis**: Redis 연동 (선택적)

### 선택적 의존성 설치

```bash
# 특정 그룹 설치
uv sync --extra dev           # 개발 의존성 포함
uv sync --extra monitoring    # 모니터링 도구 포함
uv sync --no-dev             # 프로덕션만

# 여러 그룹 동시 설치
uv sync --extra dev --extra monitoring
```

## 🧪 테스트 및 품질 관리

### 테스트 실행

```bash
make test              # 기본 테스트
make test-cov          # 커버리지 포함
uv run pytest -v      # 상세 출력
uv run pytest tests/  # 특정 디렉토리만
```

### 코드 품질

```bash
# 린팅
make lint              # 문제 확인
make lint-fix          # 자동 수정

# 포맷팅
make format            # Black + Ruff

# 타입 체크
make type-check        # MyPy 실행
```

## 🐳 Docker 빌드

UV를 사용한 멀티스테이지 Docker 빌드:

```bash
# 이미지 빌드
make docker-build

# 컨테이너 실행
make docker-run

# 또는 직접
docker build -t rag-chatbot .
docker run -p 8000:8000 rag-chatbot
```

## 🔄 마이그레이션 가이드

### pip에서 UV로 전환

1. **기존 requirements.txt 백업**
   ```bash
   mv requirements.txt requirements.txt.bak
   ```

2. **pyproject.toml 생성**
   - 이미 생성되어 있음

3. **가상환경 재생성**
   ```bash
   rm -rf venv/
   uv venv
   uv sync
   ```

4. **의존성 확인**
   ```bash
   uv pip list
   make info
   ```

## 📊 성능 비교

| 작업 | pip | uv | 개선도 |
|------|-----|----|----|
| 패키지 설치 | 45초 | 2초 | 22.5x |
| 의존성 해결 | 12초 | 0.3초 | 40x |
| 가상환경 생성 | 3초 | 0.1초 | 30x |

## 🛠️ 개발 팁

### 1. 빠른 반복 개발

```bash
# 자동 리로드로 개발 서버 실행
make dev-reload

# 또는
uv run uvicorn main:app --reload
```

### 2. 의존성 업데이트

```bash
# 모든 패키지 최신 버전으로
make update

# 특정 패키지만
uv add "fastapi@latest"
```

### 3. 환경 분리

```bash
# 다른 Python 버전 사용
echo "3.12" > .python-version
uv venv --python 3.12

# 프로젝트별 설정
uv venv --project ./my-project
```

## 🔍 트러블슈팅

### 일반적인 문제

1. **UV가 인식되지 않음**
   ```bash
   # PATH에 추가
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **의존성 충돌**
   ```bash
   # lock 파일 재생성
   rm uv.lock
   uv lock
   ```

3. **가상환경 문제**
   ```bash
   # 가상환경 재생성
   rm -rf .venv/
   uv venv
   uv sync
   ```

## 📚 참고 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [pyproject.toml 가이드](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PEP 621 - Metadata](https://peps.python.org/pep-0621/)

## ✨ 다음 단계

이제 다음과 같은 고급 기능을 활용할 수 있습니다:

1. **CI/CD 파이프라인**에 UV 통합
2. **Docker 이미지** 최적화
3. **의존성 보안 스캔** 자동화
4. **성능 모니터링** 구축

UV를 사용하면 개발 생산성이 크게 향상됩니다! 🚀
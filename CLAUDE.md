# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Backend Development (Primary)
- **Start development server**: `make dev`
- **Start with auto-reload**: `make dev-reload`
- **Start production server**: `make run`
- **Setup development environment**: `make setup`

### Development Workflow (UV + Makefile)
- **Show all commands**: `make help`
- **Install dependencies**: `make install-dev`
- **Update dependencies**: `make update`
- **Sync with lock file**: `make sync`

### Code Quality
- **Lint code**: `make lint`
- **Fix linting issues**: `make lint-fix`
- **Format code**: `make format`
- **Type checking**: `make type-check`

### Testing
- **Run tests**: `make test`
- **Run basic tests**: `make test-basic` (타임아웃 방지)
- **Run tests with coverage**: `make test-cov`
- **Test environment check**: `make test-env-check`

### Docker & Deployment
- **Build Docker image**: `make docker-build`
- **Run Docker container**: `make docker-run`
- **Clean up**: `make clean`

### Frontend Development (Optional)
- **Start web dashboard**: `cd web && npm run dev`
- **Build web dashboard**: `cd web && npm run build`
- **Start chat widget**: `cd widget && npm run dev`
- **Build widget**: `cd widget && npm run build:prod`

## Architecture Overview

This is a modular Korean RAG (Retrieval-Augmented Generation) chatbot system with a Python FastAPI backend and optional frontend components.

### 1. Python FastAPI Backend (루트 디렉토리)
- **Entry point**: `main.py` - FastAPI application with lifespan management
- **API Layer** (`/app/api/`):
  - `chat.py` - Handles chat messages with session management
  - `upload.py` - Processes document uploads (PDF, TXT, Word, Excel, CSV)
  - `documents.py` - Document management endpoints
  - `admin.py` - Admin endpoints for system management
  - `health.py` - Health checks and system stats
  - `prompts.py` - Prompt management endpoints
- **Module System** (`/app/modules/`):
  - `document_processing.py` - Document loading, splitting, and embedding
  - `retrieval_rerank.py` - Hybrid search (dense+sparse) and reranking
  - `generation.py` - Response generation using multi-LLM support
  - `enhanced_session.py` - Enhanced conversation context management
  - `gemini_embeddings.py` - Google Gemini embedding module
  - `prompt_manager.py` - Dynamic prompt management
  - `query_expansion.py` - Query expansion and enhancement
  - `session.py` - Basic session management
- **Configuration**: YAML-based configuration in `/app/config/`
- **Libraries** (`/app/lib/`): 
  - `config_loader.py` - Configuration loading and validation
  - `logger.py` - Structured logging setup
- **Models** (`/app/models/`):
  - `prompts.py` - Prompt data models
- **Vector Database**: Qdrant for storing and searching embeddings
- **Package Management**: UV for ultra-fast dependency management

## Key Technical Details

### Document Processing Pipeline
1. **Loaders**: Support for PDF, TXT, Word, Excel, HTML, Markdown, CSV
2. **Splitters**: Recursive (400 char chunks, 50 char overlap), Semantic, Markdown-aware
3. **Embeddings**: Google text-embedding-004 model
4. **Storage**: Qdrant with dense and sparse vectors

### Search Architecture
- **Hybrid Search**: 60% dense (semantic) + 40% sparse (keyword)
- **RRF Fusion**: Reciprocal Rank Fusion for combining results
- **Reranking**: Multiple options (Jina, Cohere, LLM-based)
- **Top-K Selection**: Returns top 15 results after reranking

### Multi-LLM Support
- **Primary**: Google Gemini 2.0 Flash Experimental
- **Fallback**: OpenAI GPT-4o, Anthropic Claude 3.5
- **Auto-failover**: Automatic switching on errors
- **Cost optimization**: Smart model selection

### Session Management
- Session-based conversation memory (last 5 exchanges)
- 1-hour TTL for session data
- Async context management

### Configuration System
- Base config in `app/config/config.yaml`
- Environment variable support
- Runtime configuration updates

### Error Handling
- Custom error classes with detailed context
- Structured logging with correlation IDs
- Graceful degradation for external services
- Comprehensive error recovery

### Monitoring
- Real-time cost tracking for API usage
- Performance metrics collection
- Health check endpoints
- Request/response logging

### Modern Development Stack
- **UV Package Management**: 10-100x faster than pip
- **Type Safety**: Full type hints with MyPy validation
- **Code Quality**: Black + Ruff for formatting and linting
- **Development Automation**: Makefile-based workflow
- **Container Support**: Multi-stage Docker builds

### Infrastructure/
- Docker support with optimized multi-stage builds
- UV-based dependency caching
- Health check integration
- Environment-based configuration

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Backend Development (Primary)
- **Start development server**: `cd backend && make dev`
- **Start with auto-reload**: `cd backend && make dev-reload`
- **Start production server**: `cd backend && make run`
- **Setup development environment**: `cd backend && make setup`

### Development Workflow (UV + Makefile)
- **Show all commands**: `cd backend && make help`
- **Install dependencies**: `cd backend && make install-dev`
- **Update dependencies**: `cd backend && make update`
- **Sync with lock file**: `cd backend && make sync`

### Code Quality
- **Lint code**: `cd backend && make lint`
- **Fix linting issues**: `cd backend && make lint-fix`
- **Format code**: `cd backend && make format`
- **Type checking**: `cd backend && make type-check`

### Testing
- **Run tests**: `cd backend && make test`
- **Run tests with coverage**: `cd backend && make test-cov`

### Docker & Deployment
- **Build Docker image**: `cd backend && make docker-build`
- **Run Docker container**: `cd backend && make docker-run`
- **Clean up**: `cd backend && make clean`

### Frontend Development (Optional)
- **Start web dashboard**: `cd web && npm run dev`
- **Build web dashboard**: `cd web && npm run build`
- **Start chat widget**: `cd widget && npm run dev`
- **Build widget**: `cd widget && npm run build:prod`

## Architecture Overview

This is a modular Korean RAG (Retrieval-Augmented Generation) chatbot system with a Python FastAPI backend and optional frontend components.

### 1. Python FastAPI Backend (`/backend/`)
- **Entry point**: `main.py` - FastAPI application with lifespan management
- **API Layer** (`/app/api/`):
  - `chat.py` - Handles chat messages with session management
  - `upload.py` - Processes document uploads (PDF, TXT, Word, Excel, CSV)
  - `admin.py` - Admin endpoints for system management
  - `health.py` - Health checks and system stats
- **Module System** (`/app/modules/`):
  - `document_processing.py` - Document loading, splitting, and embedding
  - `retrieval_rerank.py` - Hybrid search (dense+sparse) and reranking
  - `generation.py` - Response generation using multi-LLM support
  - `session.py` - Conversation context management
- **Configuration**: YAML-based configuration in `/app/config/`
- **Vector Database**: Qdrant for storing and searching embeddings
- **Package Management**: UV for ultra-fast dependency management

### 2. React Web Dashboard (`/web/`)
- Admin dashboard with tabs for chat, documents, stats, and uploads
- Uses Material-UI components
- TypeScript-based with Vite bundler

### 3. Chat Widget (`/widget/`)
- Embeddable chat widget for external websites
- React + TypeScript with WebSocket support
- Built as a standalone JavaScript file

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

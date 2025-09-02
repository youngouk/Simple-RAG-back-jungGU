# Multi-stage build for optimized Python RAG Chatbot with uv
FROM python:3.11-slim as builder

# Install uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY .python-version .
COPY README.md .

# Create virtual environment and install dependencies
RUN uv venv && \
    uv sync --no-dev

# Production stage
FROM python:3.11-slim

# Cache buster to force rebuild
ARG CACHE_BUST=2025-08-30-22-15
RUN echo "Cache bust: $CACHE_BUST"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Fix permissions for virtual environment executables
RUN chmod +x /app/.venv/bin/*

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs uploads/temp && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\nexec /app/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > /app/start.sh && \
    chmod +x /app/start.sh && \
    echo "Startup script created:" && cat /app/start.sh && \
    echo "Script permissions:" && ls -la /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
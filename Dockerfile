# Simple and reliable Python FastAPI build
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies using pip (avoid UV complications)
RUN python -m pip install --upgrade pip && \
    pip install -e .

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs uploads/temp && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use bash to handle dynamic PORT environment variable
CMD ["bash", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
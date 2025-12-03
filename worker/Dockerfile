FROM python:3.11-slim

LABEL maintainer="IIoT Data Quality Team"
LABEL description="Data Quality Assessment Worker for IoT Sensor Data"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv by downloading binary directly (more reliable)
RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | \
    tar -xz -C /tmp && \
    find /tmp -name "uv" -type f -executable -exec mv {} /usr/local/bin/uv \; && \
    chmod +x /usr/local/bin/uv && \
    rm -rf /tmp/uv* && \
    uv --version

# Copy pyproject.toml and source code
COPY worker/pyproject.toml /app/pyproject.toml
COPY worker /app/worker
COPY backend/config /app/config

# Install Python dependencies using uv (extract from pyproject.toml, install without package)
RUN cd /app && python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); deps=d['project']['dependencies']; import subprocess; subprocess.run(['uv', 'pip', 'install', '--system', '--no-cache'] + deps, check=True)"

# Create necessary directories
RUN mkdir -p /app/logs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run the worker
CMD ["python", "-m", "worker.src.main"]


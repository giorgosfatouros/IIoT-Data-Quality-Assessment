FROM python:3.11-slim

# Set environment variables for PostgreSQL/TimescaleDB
ENV DB_HOST=timescaledb
ENV DB_USER=iiot_user
ENV DB_PASS=iiot_password
ENV DB_PORT=5432
ENV DB_NAME=iiot_dqa
ENV PYTHONPATH=/app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package management
RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | \
    tar -xz -C /tmp && \
    find /tmp -name "uv" -type f -executable -exec mv {} /usr/local/bin/uv \; && \
    chmod +x /usr/local/bin/uv && \
    rm -rf /tmp/uv* && \
    uv --version

# Copy pyproject.toml and source code
COPY backend/pyproject.toml /app/pyproject.toml
COPY backend/ /app

# Install Python dependencies using uv (extract from pyproject.toml, install without package)
RUN cd /app && python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); deps=d['project']['dependencies']; import subprocess; subprocess.run(['uv', 'pip', 'install', '--system', '--no-cache'] + deps, check=True)"

# Create a startup script to handle environment variables
RUN echo '#!/bin/bash\n\
# Wait for PostgreSQL/TimescaleDB database to be ready\n\
echo "Waiting for TimescaleDB database..."\n\
until PGPASSWORD=${DB_PASS} psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME} -c "SELECT 1" > /dev/null 2>&1; do\n\
  echo "Database not ready, waiting..."\n\
  sleep 2\n\
done\n\
echo "Database is ready!"\n\
\n\
# Start the application\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

# Use the startup script
CMD ["/app/start.sh"]



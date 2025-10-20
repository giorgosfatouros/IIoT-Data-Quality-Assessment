FROM python:3.8-slim-bullseye

# Set environment variables for LeanXcale
ENV DB_USER=app
ENV DB_PASS=app
ENV DB_IP=127.0.0.1
ENV DB_PORT=1529
ENV DB_NAME=MOH
ENV PYTHONPATH=/app

# Install system dependencies for LeanXcale
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy LeanXcale wheel files from backend/tmp directory
COPY backend/tmp/pyLeanxcale-1.9.13-py3-none-any.whl /tmp/

# Install LeanXcale client with proper dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir /tmp/pyLeanxcale-1.9.13-py3-none-any.whl

# Install backend requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source code
COPY backend/ /app

# Create a startup script to handle environment variables
RUN echo '#!/bin/bash\n\
# Wait for LeanXcale database to be ready\n\
echo "Waiting for LeanXcale database..."\n\
while ! nc -z $DB_IP $DB_PORT; do\n\
  echo "Database not ready, waiting..."\n\
  sleep 2\n\
done\n\
echo "Database is ready!"\n\
\n\
# Start the application\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n\
' > /app/start.sh && chmod +x /app/start.sh

# Install netcat for database connectivity check
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Use the startup script
CMD ["/app/start.sh"]



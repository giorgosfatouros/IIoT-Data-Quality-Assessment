FROM python:3.8-slim-bullseye

WORKDIR /app

# Copy LeanXcale wheel from repo root using build context at project root
COPY pyLeanxcale-1.9.13_latest-py3-none-any.whl /tmp/pyLeanxcale-1.9.13_latest-py3-none-any.whl

# Install LeanXcale client first to ensure SQLAlchemy compatibility
RUN pip install --no-cache-dir /tmp/pyLeanxcale-1.9.13_latest-py3-none-any.whl

# Install backend requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source
COPY backend/ /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



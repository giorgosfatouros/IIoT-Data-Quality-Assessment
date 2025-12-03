# IIoT Data Quality Backend API

FastAPI backend for analyzing and assessing the quality of high-frequency IIoT sensor data using PostgreSQL with TimescaleDB.

## Quick Start (Docker Compose)

The recommended way to run the backend is using Docker Compose:

```bash
# From project root
./start-dev.sh
```

This starts:
- **TimescaleDB** (PostgreSQL with time-series extension)
- **Worker** (background data aggregation service)
- **Backend API** (FastAPI server on port 8000)

The backend will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Local Development Setup

For local development without Docker:

### Prerequisites
- Python 3.11+
- PostgreSQL with TimescaleDB extension
- `uv` package manager ([install](https://github.com/astral-sh/uv))

### Setup

1. **Create virtual environment and install dependencies**:
```bash
cd backend
uv sync
source .venv/bin/activate
```

3. **Configure environment variables**:
Create a `.env` file or export:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=iiot_dqa
export DB_USER=iiot_user
export DB_PASS=iiot_password
export OPENAI_API_KEY=your_key_here
export PYTHONPATH=$(pwd)
```

4. **Run the server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Backend Structure

```
backend/
├── api/              # API route handlers (FastAPI routers)
│   ├── agent_chat.py
│   ├── data_import.py
│   └── visualization.py
├── core/             # Business logic and services
│   ├── analytics.py
│   ├── connection_manager.py
│   └── import_service.py
├── models/           # Database models and connection
│   └── database.py   # SQLAlchemy engine and connection management
├── schemas/          # Pydantic schemas for request/response validation
│   ├── data_import.py
│   └── visualization.py
├── config/           # Configuration files
└── main.py           # FastAPI application entry point
```

## API Endpoints

### Core
- `GET /health` - Health check with database status
- `GET /tables` - List available machine groups
- `GET /tables/{machine_group}` - Get sensors for a machine group

### Data
- `GET /data?table={machine_group}&limit={n}` - Get aggregated insights data
- `GET /data/preprocessed?table={machine_group}&limit={n}` - Get preprocessed sensor data
- `GET /tags?table={machine_group}` - Get sensor metadata/thresholds
- `GET /sensors?table={machine_group}` - Get list of sensors

### Analytics
- `GET /analytics/aggregation_frequency?table={machine_group}` - Get aggregation frequency
- `GET /analytics/missing?table={machine_group}` - Get missing values analysis
- `GET /analytics/visualization` - Comprehensive visualization analytics

### Data Import
- `POST /import/upload` - Upload and import sensor data files
- `GET /import/status/{job_id}` - Check import job status

### DQA Agent
- `POST /agent/chat` - Chat with AI-powered data quality assessment agent

## Testing

```bash
# Health check
curl http://localhost:8000/health

# List machine groups
curl http://localhost:8000/tables

# Get sensors for a machine group
curl "http://localhost:8000/tables/MACHINE_GROUP_1"
```

## Architecture

- **FastAPI** with Pydantic v2 for request/response validation
- **SQLAlchemy 2.0+** with connection pooling for database operations
- **PostgreSQL/TimescaleDB** for time-series data storage
- **Modular structure**: Separation of API routes, business logic, and data access
- **Pandas** for data processing and analytics

## Troubleshooting

### Database Connection
- Ensure TimescaleDB is running: `docker-compose ps timescaledb`
- Check connection string in environment variables
- Verify database exists and user has proper permissions

### Server Issues
- **Port in use**: `lsof -ti tcp:8000 | xargs kill -9`
- **Module not found**: Ensure `PYTHONPATH` includes backend directory
- **Hot-reload not working**: Check that code volumes are mounted in docker-compose.yml

### Dependencies
- All dependencies are defined in `pyproject.toml`
- Use `uv sync` to create virtual environment and install dependencies

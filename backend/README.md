# IIoT Data Quality Backend API

This README describes how to deploy and run the FastAPI backend locally against a LeanXcale database running in Docker.

## DQA Agent

The backend includes an AI-powered Data Quality Assessment Agent that uses OpenAI's Agents SDK to answer questions about your sensor data. See [QUICKSTART_DQA.md](../QUICKSTART_DQA.md) and [DQA_AGENT_SETUP.md](../DQA_AGENT_SETUP.md) for setup instructions.

## Prerequisites
- Python 3.9+ installed on your machine
- LeanXcale DB running as a Docker container (exposing port 1529)
- The LeanXcale client wheel available locally at `backend/tmp/pyLeanxcale-1.9.13_latest-py3-none-any.whl`
- Optional: `uv` (fast Python package manager) for creating a virtualenv

## 1) Create and activate a virtual environment
From the project root or from `backend/`:

Using uv (recommended):
```bash
cd backend
uv venv .venv-312 --seed
source .venv-312/bin/activate
```

Using the built-in venv:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies
Install Python dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Install the LeanXcale client from the local wheel:
```bash
# from backend/
pip install --no-deps ./tmp/pyLeanxcale-1.9.13_latest-py3-none-any.whl
```
If pip warns about the wheel filename being “not correctly normalised”, you can rename it and retry:
```bash
cp ./tmp/pyLeanxcale-1.9.13_latest-py3-none-any.whl ./tmp/pyLeanxcale-1.9.13-py3-none-any.whl
pip install --no-deps ./tmp/pyLeanxcale-1.9.13-py3-none-any.whl
```

Install the client’s runtime requirements (as specified by the wheel metadata):
```bash
pip install future==0.18.2 protobuf==4.21.0 datetime==4.3 wheel==0.37.0
```

Note: We use the HTTP SQLAlchemy dialect bundled in the wheel (`pyLeanxcale`). No ODBC driver setup is needed.

## 3) Configure environment variables
Set these variables to point to your LeanXcale Docker container:
```bash
export DB_USER=app
export DB_PASS=app
export DB_IP=127.0.0.1
export DB_PORT=1529        # LeanXcale Query Engine port
export DB_NAME=MOH
export PYTHONPATH=/path/to/your/backend:$PYTHONPATH
```

The backend builds a connection URL like:
`leanxcale://<DB_USER>:<DB_PASS>@<DB_IP>:<DB_PORT>/<DB_NAME>?autocommit=False&parallel=True&txn_mode=NO_CONFLICTS_NO_LOGGING`

## 4) Start LeanXcale Database (if not running)
Ensure your LeanXcale container is running and accessible on port 1529:
```bash
# Check if container is running
docker ps | grep leanxcale

# If not running, start it (example command)
docker run --name leanxcaledb-service --env KVPEXTERNALIP='leanxcaledb-service!9800' -p 0.0.0.0:1529:1529 -d ferrari
```

## 5) Run the API server
From `backend/` with the virtualenv activated and environment variables set:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run everything in one command:
```bash
cd /path/to/your/backend && \
source .venv-312/bin/activate && \
export DB_USER=app DB_PASS=app DB_IP=127.0.0.1 DB_PORT=1529 DB_NAME=MOH && \
export PYTHONPATH=$(pwd):$PYTHONPATH && \
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 6) Test the API
Test basic endpoints to ensure everything is working:
```bash
# Health check
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok"}

# List available tables
curl http://127.0.0.1:8000/tables
# Expected: ["K3301","K3301_HOURS","KT2201","KT2201_HOURS"]

# Get sensor tags for a specific table
curl "http://127.0.0.1:8000/tags?table=K3301_HOURS"
# Expected: JSON with sensor metadata

# Get preprocessed data sample
curl "http://127.0.0.1:8000/data/preprocessed?table=K3301_HOURS&limit=5"
# Expected: JSON with data rows and columns
```

## 7) API Documentation
Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Available Endpoints

The API provides the following main endpoints:

### Core Endpoints
- `GET /health` - Health check
- `GET /tables` - List available database tables
- `GET /tables/{table_name}` - Get columns for a specific table

### Data Endpoints  
- `GET /data` - Get raw data from a table
- `GET /data/preprocessed` - Get preprocessed data with statistics
- `GET /tags` - Get sensor metadata/tags (optionally filtered by table)

### Analytics Endpoints
- `GET /analytics/aggregation_frequency` - Get data aggregation frequency analysis
- `GET /analytics/missing` - Get missing values analysis
- `GET /analytics/visualization` - Get comprehensive visualization analytics (summary stats, correlation, time series, histograms, etc.)

## Troubleshooting

### Database Connection Issues
- **Connection refused**: Ensure LeanXcale container is running and port 1529 is accessible
- **Can't load plugin**: Make sure `DB_PORT=1529` (not 8765) and the `leanxcale://` dialect is used
- **ODBC errors**: Uninstall conflicting drivers: `pip uninstall -y lxdbapi`

### Dependency Issues
- **Protobuf conflicts**: Install the expected version: `pip install protobuf==4.21.0`
- **Wheel filename issues**: Rename the wheel file to remove special characters
- **Missing pandas**: Ensure all dependencies are installed in the virtual environment

### Server Issues
- **Address already in use**: Kill existing processes: `lsof -ti tcp:8000 | xargs kill -9`
- **Module not found**: Ensure `PYTHONPATH` includes the backend directory
- **Permission errors**: Check file permissions and virtual environment activation

## Architecture Notes
- Uses FastAPI with Pydantic v2 for request/response validation
- Modular structure: `api/`, `schemas/`, `models/`, `core/` for better maintainability  
- LeanXcale HTTP dialect via `pyLeanxcale` wheel (no ODBC setup required)
- SQLAlchemy for database operations with connection pooling
- Pandas integration for data processing and analytics

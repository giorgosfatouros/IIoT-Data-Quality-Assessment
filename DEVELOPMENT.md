# Development Guide

## Quick Start

### Option 1: Full Docker Setup (Recommended for Backend)
```bash
# Start everything with one command
./start-dev.sh
```

### Option 2: Manual Setup

#### 1. Backend + Database (Docker)
```bash
# Start LeanXcale database and backend API
docker-compose up -d leanxcale-db backend

# Check if services are running
curl http://localhost:8000/health
```

#### 2. Frontend (Local Node.js)
```bash
cd frontend
npm install
npm run dev
```

## Development Workflow

### Backend Development
- **Docker**: Backend runs in Docker with LeanXcale database
- **Hot Reload**: Backend supports hot reload in development
- **Logs**: `docker-compose logs -f backend`

### Frontend Development  
- **Local**: Frontend runs locally with Node.js/Vite
- **Hot Reload**: Vite provides instant hot reload
- **Debugging**: Full access to browser dev tools

## Environment Setup

1. **Copy environment file**:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env`** and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Start services**:
   ```bash
   ./start-dev.sh
   ```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | React/Vite development server |
| Backend API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| LeanXcale DB | localhost:1529 | Database (internal) |

## Troubleshooting

### LeanXcale Connection Issues
```bash
# Check database logs
docker-compose logs leanxcale-db

# Test connection
docker-compose exec backend python -c "from models.database import get_engine; print(get_engine())"
```

### Frontend Issues
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Backend Issues
```bash
# Rebuild backend container
docker-compose build backend
docker-compose up -d backend
```

## Production Deployment

For production, uncomment the frontend service in `docker-compose.yml` and use:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#!/bin/bash

# FAME Data Quality Assessment - Development Startup Script
# This script sets up the development environment with Docker

set -e

echo "🚀 Starting FAME Data Quality Assessment Development Environment"
echo "================================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp env.example .env
    echo "⚠️  Please edit .env file and add your OpenAI API key"
    echo "   Then run this script again."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo "🔧 Environment Configuration:"
echo "   DB_IP: $DB_IP"
echo "   DB_PORT: $DB_PORT"
echo "   DB_NAME: $DB_NAME"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# Build and start backend services (database + backend API)
echo "🏗️  Building and starting backend services..."
docker-compose up --build -d leanxcale-db backend

echo "⏳ Waiting for services to be ready..."

# Wait for LeanXcale database
echo "   Waiting for LeanXcale database..."
timeout=60
while ! docker-compose exec -T leanxcale-db nc -z localhost 1529 2>/dev/null; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ LeanXcale database failed to start within 60 seconds"
        docker-compose logs leanxcale-db
        exit 1
    fi
done
echo "   ✅ LeanXcale database is ready"

# Wait for backend
echo "   Waiting for backend service..."
timeout=60
while ! curl -f http://localhost:8000/health 2>/dev/null; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ Backend service failed to start within 60 seconds"
        docker-compose logs backend
        exit 1
    fi
done
echo "   ✅ Backend service is ready"

# Start frontend locally
echo "   Starting frontend locally..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "   Installing frontend dependencies..."
    npm install
fi

echo "   Starting Vite development server..."
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend
echo "   Waiting for frontend service..."
timeout=30
while ! curl -f http://localhost:5173 2>/dev/null; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ Frontend service failed to start within 30 seconds"
        kill $FRONTEND_PID 2>/dev/null
        exit 1
    fi
done
echo "   ✅ Frontend service is ready"

echo ""
echo "🎉 All services are running!"
echo "================================================================"
echo "📊 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🗄️  LeanXcale DB: localhost:1529"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo "================================================================"

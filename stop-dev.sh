#!/bin/bash

echo "🛑 Stopping FAME Data Quality Assessment Development Environment"
echo "================================================================"

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose down

# Kill any remaining frontend processes
echo "🖥️  Stopping frontend processes..."
pkill -f "vite.*5173" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

echo "✅ All services stopped!"
echo "================================================================"

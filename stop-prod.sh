#!/bin/bash

# FAME Data Quality Assessment - Production Stop Script
# This script stops all production services

echo "🛑 Stopping FAME Data Quality Assessment Production Environment"
echo "================================================================"

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose down

echo "✅ All services stopped!"
echo "================================================================"


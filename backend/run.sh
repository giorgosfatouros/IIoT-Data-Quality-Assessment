#!/bin/sh

# Fetch the IP address of the running LeanXscale container
export DB_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' leanxcaledb-service)

# Set other environment variables if needed
export DB_USER=app
export DB_PASS=app
export DB_PORT=1529
export DB_NAME=MOH
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

# Run Docker Compose
docker-compose up


docker push gfatouros/fame-data-quality:latest

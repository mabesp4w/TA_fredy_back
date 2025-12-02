#!/bin/bash

# Script to rebuild Docker container with updated dependencies

echo "Stopping existing containers..."
docker compose down

echo "Removing old images..."
docker compose down --rmi all

echo "Building new Docker image with updated dependencies..."
docker compose build --no-cache

echo "Starting containers..."
docker compose up -d

echo "Docker rebuild completed!"
echo "You can now test the prediction API at http://localhost:8103/api/prediction/"

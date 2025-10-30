#!/bin/bash
# Docker cleanup script to prevent disk space accumulation
# Run this periodically to clean up old images, containers, and build cache

set -e

echo "=== Docker Cleanup Script ==="
echo "Starting cleanup process..."

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f

# Remove unused build cache
echo "Removing build cache..."
docker builder prune -f

# Remove dangling volumes (CAREFUL: only removes unused volumes)
echo "Removing dangling volumes..."
docker volume prune -f

echo ""
echo "=== Cleanup Summary ==="
docker system df

echo ""
echo "Cleanup complete!"

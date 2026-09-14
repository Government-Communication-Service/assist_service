#!/usr/bin/env bash
# Build and push the app image to the preview ECR repo.
#
# Usage (from repo root):
#   infra/scripts/image-build-push.sh --tag IMAGE_TAG
#
# Builds for linux/amd64 explicitly so the image runs on the ECS EC2 instance
# regardless of whether you're building on an Apple Silicon (arm64) Mac.
set -euo pipefail

TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --tag) TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$TAG" ]] && { echo "ERROR: --tag is required"; exit 1; }

REGION="${AWS_DEFAULT_REGION:-eu-west-2}"
REPO_NAME="copilot-api"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "Building $REPO:$TAG for linux/amd64..."
docker build --platform linux/amd64 -f Dockerfile.ecs -t "$REPO:$TAG" .

echo "Pushing $REPO:$TAG..."
docker push "$REPO:$TAG"

echo "Done: $REPO:$TAG"

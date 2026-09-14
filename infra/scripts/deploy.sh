#!/usr/bin/env bash
# Registers a new task-def revision for the preview ECS service with a new
# image tag, and forces a redeploy. Does not touch CDK.
#
# Usage:
#   infra/scripts/deploy.sh --tag IMAGE_TAG
#
# Requires: aws cli, jq. The "preview-copilot-api" task family and
# "assist-preview" cluster must already exist (`cdk deploy PreviewFoundation`
# then `cdk deploy PreviewService --parameters ImageTag=<tag>` in infra/).
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
FAMILY="preview-copilot-api"
CLUSTER="assist-preview"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/copilot-api:$TAG"

echo "Looking up the preview service on cluster $CLUSTER..."
SERVICE=$(aws ecs list-services --region "$REGION" --cluster "$CLUSTER" --query 'serviceArns[0]' --output text)
[[ -z "$SERVICE" || "$SERVICE" == "None" ]] && { echo "ERROR: no service found on cluster $CLUSTER — deploy PreviewFoundation then PreviewService in infra/ first"; exit 1; }

echo "Reading current task definition ($FAMILY)..."
CURRENT=$(aws ecs describe-task-definition --task-definition "$FAMILY" --region "$REGION" --query 'taskDefinition')

NEW_DEF=$(echo "$CURRENT" | jq --arg IMAGE "$IMAGE" '
  .containerDefinitions[0].image = $IMAGE
  | del(
      .taskDefinitionArn, .revision, .status, .requiresAttributes,
      .compatibilities, .registeredAt, .registeredBy
    )
')

echo "Registering new revision with image $IMAGE..."
NEW_ARN=$(aws ecs register-task-definition --region "$REGION" --cli-input-json "$NEW_DEF" \
  --query 'taskDefinition.taskDefinitionArn' --output text)

echo "Updating service $SERVICE -> $NEW_ARN..."
aws ecs update-service --region "$REGION" --cluster "$CLUSTER" --service "$SERVICE" \
  --task-definition "$NEW_ARN" --force-new-deployment > /dev/null

echo "Done. Watch rollout with:"
echo "  aws ecs describe-services --region $REGION --cluster $CLUSTER --services $SERVICE --query 'services[0].deployments'"

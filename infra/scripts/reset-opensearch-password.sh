#!/usr/bin/env bash
# Re-syncs the OpenSearch domain's admin password to match what the app
# already expects (copilot_preview_secrets' opensearch_password field).
#
# Why this exists: after the domain's password falls out of sync (e.g.
# after an instance crash/replacement), submitting the *same* value the app
# already expects via `update-domain-config` is a silent no-op - AWS diffs
# against the last value it actually applied, so nothing changes and auth
# keeps failing. Empirically, the fix is to change the password to some
# other value first (forcing a real update), then change it back to the
# value in the secret (also now a real update, since it differs from the
# throwaway). This script does that two-step dance.
#
# Usage:
#   infra/scripts/reset-opensearch-password.sh
#
# Requires: aws cli, jq, openssl. Does not touch CDK or restart the ECS
# service - force a new deployment separately if the app's OpenSearch
# client needs to pick this up (see infra/README.md).
set -euo pipefail

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 1; }

REGION="${AWS_DEFAULT_REGION:-eu-west-2}"
DOMAIN_NAME="preview-assist-opensearch"
SECRET_ID="copilot_preview_secrets"

wait_for_active() {
  echo "Waiting for AdvancedSecurityOptions to go Active..."
  for _ in $(seq 1 60); do
    STATE=$(aws opensearch describe-domain-config --region "$REGION" --domain-name "$DOMAIN_NAME" \
      --query 'DomainConfig.AdvancedSecurityOptions.Status.State' --output text)
    [[ "$STATE" == "Active" ]] && return 0
    sleep 10
  done
  echo "ERROR: timed out waiting for Active (last state: $STATE)" >&2
  exit 1
}

set_master_password() {
  local password="$1"
  jq -n --arg name "$DOMAIN_NAME" --arg pw "$password" '{
    DomainName: $name,
    AdvancedSecurityOptions: {
      Enabled: true,
      InternalUserDatabaseEnabled: true,
      MasterUserOptions: { MasterUserName: "admin", MasterUserPassword: $pw }
    }
  }' > /tmp/opensearch-master-password-update.json
  aws opensearch update-domain-config --region "$REGION" \
    --cli-input-json file:///tmp/opensearch-master-password-update.json > /dev/null
  rm -f /tmp/opensearch-master-password-update.json
  wait_for_active
}

echo "Reading opensearch_password from $SECRET_ID..."
REAL_PASSWORD=$(aws secretsmanager get-secret-value --region "$REGION" --secret-id "$SECRET_ID" \
  --query SecretString --output text | jq -r '.opensearch_password')
[[ -z "$REAL_PASSWORD" || "$REAL_PASSWORD" == "null" ]] && { echo "ERROR: opensearch_password missing from $SECRET_ID" >&2; exit 1; }

# Must satisfy OpenSearch's own complexity policy (upper, lower, digit, symbol).
THROWAWAY_PASSWORD="Rt9-$(openssl rand -hex 18)"

echo "Setting a throwaway password to force a real update..."
set_master_password "$THROWAWAY_PASSWORD"

echo "Setting the password back to the value in $SECRET_ID..."
set_master_password "$REAL_PASSWORD"

echo "Done. The domain's admin password now matches $SECRET_ID."
echo "If the app is still failing to authenticate, force a new ECS deployment so it re-fetches the secret:"
echo "  aws ecs update-service --cluster assist-preview --service <service-arn> --force-new-deployment"



### Useful snippets on the ECS task
# Follow the instruction on README.md to access the task
#
# PW=$(python -c "from app.config import settings; print(settings.opensearch_password.get_secret_value())")
# curl -skv -u "admin:$PW" -o /dev/null -w "status: %{http_code}\n"   "https://${OPENSEARCH_HOST}:${OPENSEARCH_PORT:-443}/_cluster/health"

# sh-5.2# python -c "
# from app.opensearch.service import create_client
# c = create_client()
# try:
#     print(c.info())
# except Exception as e:
#     print(type(e).__name__, e)
# "

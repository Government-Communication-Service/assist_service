#!/usr/bin/env bash
# Fetch the real AWS values a developer needs in infra/cdk.context.json,
# looking each one up by the human-readable name/tag it's known by rather
# than requiring anyone to hunt for IDs/ARNs in the console.
#
# Usage:
#   export AWS_PROFILE=<your AWS profile>
#   infra/scripts/fetch-context-values.sh                    # print JSON to stdout
#   infra/scripts/fetch-context-values.sh > infra/cdk.context.json
#
# A couple of fields can't be looked up this way and are printed as-is:
# - postgres_user / bastion_name_tag / app_secret_name are naming
#   conventions this project chose, not values to discover.
# - gcs_data_api_url comes from the dev Elastic Beanstalk environment's
#   config. This script tries to find that environment automatically; if it
#   can't find exactly one match, it leaves the value null and prints a
#   warning so you know to fill it in by hand.
set -euo pipefail

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 1; }

REGION="${AWS_DEFAULT_REGION:-eu-west-2}"

# --- names/tags this environment is known by - not secrets, just the
# lookup keys used to find the real IDs/ARNs below ------------------------
VPC_NAME_TAG="GCS-LLM"
RDS_INSTANCE_ID="gcs-llm-db"
BASTION_NAME_TAG="gcs-llm-db-prod-bastion"
PREFIX_LIST_NAME="copilot-allowed-ip-addresses"
WAF_NAME="dev-gcs-copilot-api-waf"
SECRETS_POLICY_NAME="getCopilotSecrets"
HOSTED_ZONE_NAME="api.copilot.gcs.civilservice.gov.uk"
POSTGRES_USER="preview_gcs_llm_copilot_user"
APP_SECRET_NAME="copilot_preview_secrets"

echo "Account/region..." >&2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

echo "VPC and subnets..." >&2
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters "Name=tag:Name,Values=$VPC_NAME_TAG" --query 'Vpcs[0].VpcId' --output text)
[[ "$VPC_ID" == "None" || -z "$VPC_ID" ]] && { echo "ERROR: no VPC tagged Name=$VPC_NAME_TAG" >&2; exit 1; }

SUBNETS_JSON=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[].{Id:SubnetId,Az:AvailabilityZone}' --output json)

# Classify each subnet by its route table: a route through an internet
# gateway means public, a route through a NAT gateway means private-with-
# egress (what ECS tasks need). This VPC also has subnet(s)/AZ(s) that are
# neither - e.g. an isolated/data tier, or a third AZ with no public subnet
# at all - which this project doesn't use and must not pick up.
SUBNETS_WITH_TYPE="[]"
while read -r id az; do
  rt=$(aws ec2 describe-route-tables --region "$REGION" \
    --filters "Name=association.subnet-id,Values=$id" --output json)
  if [[ "$(jq '.RouteTables | length' <<<"$rt")" -eq 0 ]]; then
    rt=$(aws ec2 describe-route-tables --region "$REGION" \
      --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.main,Values=true" --output json)
  fi
  is_public=$(jq '[.RouteTables[].Routes[]?.GatewayId // ""] | any(startswith("igw-"))' <<<"$rt")
  has_nat=$(jq '[.RouteTables[].Routes[]?.NatGatewayId // ""] | any(. != "")' <<<"$rt")
  SUBNETS_WITH_TYPE=$(jq --arg id "$id" --arg az "$az" --argjson pub "$is_public" --argjson nat "$has_nat" \
    '. + [{id:$id, az:$az, public:$pub, private_with_nat:$nat}]' <<<"$SUBNETS_WITH_TYPE")
done < <(jq -r '.[] | "\(.Id) \(.Az)"' <<<"$SUBNETS_JSON")

# Only AZs with a public subnet are usable - the ALB needs a matching public
# subnet in every AZ tasks run in, so this is what actually pins the AZ set
# to the 2 the ALB/ECS pair uses, dropping any extra tier/AZ automatically.
AVAILABILITY_ZONES=$(jq -c '[.[] | select(.public==true) | .az] | unique | sort' <<<"$SUBNETS_WITH_TYPE")
PUBLIC_SUBNET_IDS=$(jq -c --argjson azs "$AVAILABILITY_ZONES" \
  '[$azs[] as $az | (map(select(.public==true and .az==$az)) | .[0].id)]' <<<"$SUBNETS_WITH_TYPE")
PRIVATE_SUBNET_IDS=$(jq -c --argjson azs "$AVAILABILITY_ZONES" \
  '[$azs[] as $az | (map(select(.private_with_nat==true and .az==$az)) | .[0].id)]' <<<"$SUBNETS_WITH_TYPE")

if jq -e 'any(.[]; . == null)' <<<"$PUBLIC_SUBNET_IDS" >/dev/null \
  || jq -e 'any(.[]; . == null)' <<<"$PRIVATE_SUBNET_IDS" >/dev/null; then
  echo "ERROR: couldn't find one public + one NAT-routed private subnet in every AZ - inspect manually:" >&2
  jq . <<<"$SUBNETS_WITH_TYPE" >&2
  exit 1
fi

echo "RDS instance..." >&2
RDS_JSON=$(aws rds describe-db-instances --region "$REGION" \
  --db-instance-identifier "$RDS_INSTANCE_ID" --output json)
RDS_ENDPOINT=$(jq -r '.DBInstances[0].Endpoint.Address' <<<"$RDS_JSON")
RDS_SECURITY_GROUP_ID=$(jq -r '.DBInstances[0].VpcSecurityGroups[0].VpcSecurityGroupId' <<<"$RDS_JSON")

echo "ALB ingress prefix list..." >&2
ALB_INGRESS_PREFIX_LIST_ID=$(aws ec2 describe-managed-prefix-lists --region "$REGION" \
  --filters "Name=prefix-list-name,Values=$PREFIX_LIST_NAME" \
  --query 'PrefixLists[0].PrefixListId' --output text)

echo "WAF web ACL..." >&2
WAF_WEB_ACL_ARN=$(aws wafv2 list-web-acls --region "$REGION" --scope REGIONAL \
  --query "WebACLs[?Name=='$WAF_NAME'].ARN | [0]" --output text)

echo "getCopilotSecrets IAM policy..." >&2
SECRETS_POLICY_ARN=$(aws iam list-policies --scope Local \
  --query "Policies[?PolicyName=='$SECRETS_POLICY_NAME'].Arn | [0]" --output text)

echo "Route53 hosted zone..." >&2
ZONE_JSON=$(aws route53 list-hosted-zones-by-name --dns-name "$HOSTED_ZONE_NAME" \
  --query 'HostedZones[0]' --output json)
HOSTED_ZONE_ID=$(jq -r '.Id' <<<"$ZONE_JSON" | sed 's#/hostedzone/##')
HOSTED_ZONE_NAME_FOUND=$(jq -r '.Name' <<<"$ZONE_JSON" | sed 's/\.$//')
if [[ "$HOSTED_ZONE_NAME_FOUND" != "$HOSTED_ZONE_NAME" ]]; then
  echo "WARNING: no hosted zone named $HOSTED_ZONE_NAME - closest match was $HOSTED_ZONE_NAME_FOUND" >&2
fi

echo "Dev Elastic Beanstalk environment (for gcs_data_api_url)..." >&2
DEV_ENVS_JSON=$(aws elasticbeanstalk describe-environments --no-include-deleted --output json \
  | jq -c '[.Environments[] | select(.EnvironmentName | test("dev"; "i")) | select(.EnvironmentName | test("copilot"; "i"))]')
DEV_ENV_COUNT=$(jq 'length' <<<"$DEV_ENVS_JSON")

GCS_DATA_API_URL="null"
if [[ "$DEV_ENV_COUNT" -eq 1 ]]; then
  APP_NAME=$(jq -r '.[0].ApplicationName' <<<"$DEV_ENVS_JSON")
  ENV_NAME=$(jq -r '.[0].EnvironmentName' <<<"$DEV_ENVS_JSON")
  CONFIG_JSON=$(aws elasticbeanstalk describe-configuration-settings \
    --application-name "$APP_NAME" --environment-name "$ENV_NAME" --output json)
  GCS_DATA_API_URL=$(jq -r '.ConfigurationSettings[0].OptionSettings[]
    | select(.Namespace=="aws:elasticbeanstalk:application:environment" and .OptionName=="GCS_DATA_API_URL")
    | .Value // "null"' <<<"$CONFIG_JSON")
  [[ -z "$GCS_DATA_API_URL" ]] && GCS_DATA_API_URL="null"
  GCS_DATA_API_URL="\"$GCS_DATA_API_URL\""
else
  echo "WARNING: found $DEV_ENV_COUNT candidate dev EB environments, expected exactly 1 - leaving gcs_data_api_url as null, fill in by hand:" >&2
  jq -r '.[].EnvironmentName' <<<"$DEV_ENVS_JSON" >&2
fi

jq -n \
  --arg account "$ACCOUNT" \
  --arg region "$REGION" \
  --arg vpc_id "$VPC_ID" \
  --argjson public_subnet_ids "$PUBLIC_SUBNET_IDS" \
  --argjson private_subnet_ids "$PRIVATE_SUBNET_IDS" \
  --argjson availability_zones "$AVAILABILITY_ZONES" \
  --arg rds_endpoint "$RDS_ENDPOINT" \
  --arg rds_security_group_id "$RDS_SECURITY_GROUP_ID" \
  --arg postgres_user "$POSTGRES_USER" \
  --arg alb_ingress_prefix_list_id "$ALB_INGRESS_PREFIX_LIST_ID" \
  --arg waf_web_acl_arn "$WAF_WEB_ACL_ARN" \
  --arg bastion_name_tag "$BASTION_NAME_TAG" \
  --arg app_secret_name "$APP_SECRET_NAME" \
  --arg secrets_policy_arn "$SECRETS_POLICY_ARN" \
  --arg hosted_zone_id "$HOSTED_ZONE_ID" \
  --arg hosted_zone_name "$HOSTED_ZONE_NAME" \
  --argjson gcs_data_api_url "$GCS_DATA_API_URL" \
  '{
    account: $account,
    region: $region,
    vpc_id: $vpc_id,
    public_subnet_ids: $public_subnet_ids,
    private_subnet_ids: $private_subnet_ids,
    availability_zones: $availability_zones,
    rds_endpoint: $rds_endpoint,
    rds_security_group_id: $rds_security_group_id,
    postgres_user: $postgres_user,
    alb_ingress_prefix_list_id: $alb_ingress_prefix_list_id,
    waf_web_acl_arn: $waf_web_acl_arn,
    bastion_name_tag: $bastion_name_tag,
    app_secret_name: $app_secret_name,
    secrets_policy_arn: $secrets_policy_arn,
    hosted_zone_id: $hosted_zone_id,
    hosted_zone_name: $hosted_zone_name,
    gcs_data_api_url: $gcs_data_api_url
  }'

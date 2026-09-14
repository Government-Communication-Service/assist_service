# Preview environment (ECS on EC2)

CDK (Python) for a preview environment, split into two stacks:

- `PreviewFoundation` — network, IAM, ECR, S3, OpenSearch, ECS cluster/EC2, secrets. No containers.
- `PreviewService` — task definition, ECS service, ALB, DNS. Depends on `PreviewFoundation` and requires an `ImageTag` parameter on every deploy.

`config.py` (pydantic-settings) is the single source of truth for every name/ID either stack uses.

There are three phases below: **1. one-time setup** (per developer machine), **2. one-time environment setup** (run once, ever, for this environment), and **3. deploying an image** (run every time there's new code to ship).

## 1. One-time setup (per developer machine)

Org policy doesn't allow AWS identifiers (account IDs, VPC/subnet/security-group IDs, ARNs, ...)
in this repo, so they live in `cdk.context.json`, which is gitignored.

```sh
cd infra
export AWS_PROFILE=<your AWS profile>
aws sts get-caller-identity   # confirm you're pointed at the right account
```

Populate `cdk.context.json` one of two ways:

```sh
# Option A: fetch the real values automatically
scripts/fetch-context-values.sh > cdk.context.json

# Option B: copy the template and fill in the values by hand (ask if you don't have them)
cp cdk.context.json.example cdk.context.json
```

Then finish setup:

```sh
uv sync

cdk bootstrap   # once ever per AWS account/region, harmless to re-run
```

## 2. One-time environment setup

Run these once, ever, to stand up the environment. Skip straight to "3. Deploying an
image" if this has already been done.

### Deploy the Foundation stack

```sh
cd infra
cdk deploy PreviewFoundation
```

Creates the ECR repo, IAM roles, S3 bucket, OpenSearch domain (~15-20 min), ECS
cluster + EC2 instance, and the generated secrets, including `copilot_preview_secrets`
(the secret the app reads at runtime).

### Create the database role

The Postgres role has to exist on the shared RDS instance before the service can
start, and CDK can't create it — it's not a CDK-managed resource.

```sh
POSTGRES_PW=$(aws secretsmanager get-secret-value --secret-id copilot-preview/postgres-password \
  --region eu-west-2 --query SecretString --output text)

ssh -t <bastion-ssh-alias> \
  "psql -h gcs-llm-db.c3osqu0y68vr.eu-west-2.rds.amazonaws.com -p 5432 -U postgres -d postgres"
```

```sql
CREATE DATABASE preview_gcs_llm_copilot;
CREATE ROLE preview_gcs_llm_copilot_user WITH LOGIN PASSWORD '<value of $POSTGRES_PW above>';
GRANT ALL PRIVILEGES ON DATABASE preview_gcs_llm_copilot TO preview_gcs_llm_copilot_user;
\c preview_gcs_llm_copilot
GRANT ALL ON SCHEMA public TO preview_gcs_llm_copilot_user;
```

### Manual steps on secrets

Share `auth_secret_key` with the frontend team.

```sh
aws secretsmanager get-secret-value --secret-id copilot-preview/auth-secret-key \
  --region eu-west-2 --query SecretString --output text
```

Send the output to the frontend team (and anyone else who needs to call this
environment) as the `Auth-Token` header value.

Add the Bugsnag API key to `copilot_preview_secrets` so that it can call Bugsnag.

## 3. Deploying an image

Run this every time there's new code to ship — including the very first deploy.

```sh
cd "$(git rev-parse --show-toplevel)"   # repo root - image-build-push.sh expects to run from here
TAG=$(git rev-parse --short HEAD)
infra/scripts/image-build-push.sh --tag "$TAG"

# Then deploy the service, pinned to that tag — one of:
infra/scripts/deploy.sh --tag "$TAG"                                    # Option A: raw ECS API calls, no CDK
cd infra && cdk deploy PreviewService --exclusively --parameters ImageTag="$TAG"  # Option B: CDK

# Check it's up
curl https://preview.api.copilot.gcs.civilservice.gov.uk/healthcheck
# -> {"status":"fine"}
```

Both options register a new task definition revision and update the service. For
the very first-ever deploy, use Option B — the service doesn't exist yet, and
Option A only updates an existing one. After that, pick either.

## Connecting to the database

```sh
scripts/db_queries/db-connect.sh <bastion-ssh-alias> preview postgres
```

This drops into an interactive session with the 'postgres' user. To enter a session with fewer permissions replace `postgres` with `preview_gcs_llm_copilot_user` and enter the password from the secret.
Requires the bastion SSH alias to already be set up (see the `ssh-setup` skill) and
the bastion EC2 instance to be running.

## Debugging the running container

```sh
aws ecs execute-command --cluster assist-preview --task <task-id> --container App \
  --interactive --command "/bin/sh"
```

## Pausing (stop paying for compute without destroying anything)

```sh
aws ecs update-service --cluster assist-preview --service <service-arn> --desired-count 0
aws autoscaling update-auto-scaling-group --auto-scaling-group-name <asg-name> \
  --min-size 0 --max-size 0 --desired-capacity 0
```

The OpenSearch domain and ALB keep running (and costing) either way — full teardown
needs `cdk destroy` (see below).

## Unpausing

Reverse order - bring the instance back before asking the service to place a task on it:

```sh
aws autoscaling update-auto-scaling-group --auto-scaling-group-name <asg-name> \
  --min-size 1 --max-size 1 --desired-capacity 1
aws ecs update-service --cluster assist-preview --service <service-arn> --desired-count 1
```

Give the new instance a minute or two to launch and register with the cluster before
the task can actually place.

## Full teardown

```sh
cd infra
cdk destroy PreviewService PreviewFoundation
```

Passing both stack names to one `cdk destroy` handles the dependency ordering
automatically. This only touches resources these two stacks created — the shared
VPC, RDS instance, bastion, and `getCopilotSecrets` policy are untouched, and the
DB role/database from step 2 aren't CDK resources either (dropping them, if ever
needed, is a manual step).

## Perimeter (matches dev/test/prod)

The ALB only accepts HTTPS from `alb_ingress_prefix_list_id` (the account's
`copilot-allowed-ip-addresses` managed prefix list), and is associated with the same
`waf_web_acl_arn` WebACL dev's ALB uses. Both are existing account resources
referenced by ID/ARN from `cdk.context.json`, not created here.

## Known assumptions (correct if wrong)

- `frontend_origin` (CORS-allowed origin, in `cdk.context.json` or left to its
  default) is `https://preview.connect.communications.gov.uk`. Update once the
  frontend team's actual subdomain is confirmed, then redeploy.
- `bugsnag_api_key` in `copilot_preview_secrets` starts as a placeholder — errors
  won't reach Bugsnag until a real project key replaces it.
- No CloudWatch alarms — Container Insights (enabled on the ECS cluster) gives
  CPU/memory visibility; add alarms if you also want alerting.

## What's NOT here yet

CodeCommit/CodePipeline wiring — deliberately last, per the original spec. Once the
above is proven stable, wire a pipeline stage in the existing `copilot-api`
CodeCommit repo to call `scripts/image-build-push.sh` + `scripts/deploy.sh` (or the
CDK equivalent) on push.

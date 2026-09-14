#!/usr/bin/env python3
"""CDK entrypoint for the copilot-api preview environment (ECS on EC2).

Deploys two stacks: `PreviewFoundation` (network/IAM/ECR/data/compute - no
containers, always safe to deploy) and `PreviewService` (task def/ECS
service/ALB/DNS, depends on Foundation and requires an `ImageTag` parameter).
Redeploying just the app image is `cdk deploy PreviewService --parameters
ImageTag=<tag>` (or `infra/scripts/deploy.sh`, pending a decision on which to
standardise on) — see `infra/README.md`.

Settings (AWS identifiers + resource names) come from `config.settings`,
loaded from `cdk.context.json` (gitignored — copy `cdk.context.json.example`
first).
"""

import subprocess

import aws_cdk as cdk
from config import settings
from preview.foundation_stack import PreviewFoundationStack
from preview.service_stack import PreviewServiceStack

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except subprocess.CalledProcessError:
    git_commit = "unknown"

app = cdk.App()
env = cdk.Environment(account=settings.account, region=settings.region)

foundation = PreviewFoundationStack(app, "PreviewFoundation", settings=settings, env=env)
PreviewServiceStack(app, "PreviewService", settings=settings, foundation=foundation, env=env)

cdk.Tags.of(app).add("project", "copilot-api-preview")
cdk.Tags.of(app).add("git-commit", git_commit)

app.synth()

"""ECR repository for the application image.

`infra/scripts/image-build-push.sh` pushes here, tagged by git sha;
`infra/scripts/deploy.sh` registers a new task-def revision pointing at the
new tag - no CDK involved in a routine redeploy.
"""

from aws_cdk import RemovalPolicy
from aws_cdk import aws_ecr as ecr
from config import Settings
from constructs import Construct


class Ecr(Construct):
    def __init__(self, scope: Construct, construct_id: str, settings: Settings) -> None:
        super().__init__(scope, construct_id)

        self.repository = ecr.Repository(
            self,
            "Repository",
            repository_name=settings.ecr_repo_name,
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[
                ecr.LifecycleRule(description="Keep only the 15 most recent images", max_image_count=15),
            ],
        )

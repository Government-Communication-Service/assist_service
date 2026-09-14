"""The container half of the preview environment: task definition, ECS
service, ALB, DNS. Depends on `PreviewFoundationStack` for the VPC security
groups, IAM roles, ECR repo, S3 bucket, OpenSearch domain, and ECS cluster/
capacity provider - none of which this stack creates itself.

Requires the `ImageTag` CloudFormation parameter (the tag pushed to the ECR
repo by `infra/scripts/image-build-push.sh`) on every deploy - there's no
default, so a deploy can't silently start (or revert) the service onto the
wrong image.
"""

from aws_cdk import CfnOutput, CfnParameter, Stack
from config import Settings
from constructs import Construct

from preview.foundation_stack import PreviewFoundationStack
from preview.service import Service


class PreviewServiceStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        settings: Settings,
        foundation: PreviewFoundationStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        image_tag = CfnParameter(
            self,
            "ImageTag",
            type="String",
            allowed_pattern=".+",
            constraint_description="ImageTag must not be empty",
            description="ECR image tag to deploy, e.g. a git short-sha pushed by image-build-push.sh",
        )

        service = Service(
            self,
            "Service",
            settings=settings,
            vpc=foundation.network.vpc,
            alb_security_group=foundation.network.alb_sg,
            task_security_group=foundation.network.task_sg,
            cluster=foundation.compute.cluster,
            capacity_provider_name=foundation.compute.capacity_provider.capacity_provider_name,
            execution_role=foundation.roles.execution_role,
            task_role=foundation.roles.task_role,
            repository=foundation.ecr.repository,
            bucket=foundation.data.bucket,
            opensearch_domain=foundation.data.domain,
            image_tag=image_tag.value_as_string,
            log_group=foundation.log_group,
        )

        CfnOutput(self, "AlbDnsName", value=service.alb.load_balancer_dns_name)

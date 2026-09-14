"""The non-container half of the preview environment: network, IAM, ECR,
data (S3 + OpenSearch), and ECS compute (cluster + EC2 capacity). Nothing
here runs a container, so this stack is always safe to deploy on its own.
"""

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_logs as logs
from config import Settings
from constructs import Construct

from preview.compute import Compute
from preview.data import Data
from preview.ecr import Ecr
from preview.generated_secrets import GeneratedSecrets
from preview.network import Network
from preview.roles import Roles


class PreviewFoundationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, settings: Settings, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.network = Network(self, "Network", settings=settings)
        self.roles = Roles(self, "Roles", settings=settings)
        self.ecr = Ecr(self, "Ecr", settings=settings)
        self.secrets = GeneratedSecrets(self, "Secrets", settings=settings)
        self.data = Data(
            self,
            "Data",
            settings=settings,
            vpc=self.network.vpc,
            opensearch_sg=self.network.opensearch_sg,
            opensearch_password=self.secrets.opensearch_password,
        )
        self.data.bucket.grant_read_write(self.roles.task_role)

        self.compute = Compute(
            self,
            "Compute",
            settings=settings,
            vpc=self.network.vpc,
            instance_security_group=self.network.task_sg,
            instance_role=self.roles.instance_role,
        )

        # Created here, not in PreviewServiceStack: the task role's log-write
        # grant (added when the container's logging driver binds to it in
        # Service) would otherwise create a cycle - Foundation's TaskRole
        # referencing a log group in the stack that depends on TaskRole.
        self.log_group = logs.LogGroup(
            self,
            "LogGroup",
            log_group_name=settings.log_group_name,
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        CfnOutput(self, "OpensearchEndpoint", value=self.data.domain.domain_endpoint)
        CfnOutput(self, "BucketName", value=self.data.bucket.bucket_name)

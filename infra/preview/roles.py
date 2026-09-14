"""IAM roles for the preview ECS/EC2 service.

S3 bucket access is granted in `stack.py` (`bucket.grant_read_write`), not
here. No OpenSearch (`es:*`) permissions on the task role: the app
authenticates to OpenSearch with HTTP basic auth (`app/opensearch/service.py`),
not SigV4.
"""

from aws_cdk import aws_iam as iam
from config import Settings
from constructs import Construct


class Roles(Construct):
    def __init__(self, scope: Construct, construct_id: str, settings: Settings) -> None:
        super().__init__(scope, construct_id)

        # --- ECS task execution role (pulls image, ships container logs) ---
        self.execution_role = iam.Role(
            self,
            "ExecutionRole",
            role_name="preview-execution-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy"),
            ],
        )

        # --- task role (the running container's own identity) --------------
        self.task_role = iam.Role(
            self,
            "TaskRole",
            role_name="preview-task-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_managed_policy_arn(self, "SecretsPolicy", settings.secrets_policy_arn),
            ],
        )
        # Bedrock - invoke Claude models (incl. cross-region inference profiles).
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=["*"],
            )
        )
        # ECS Exec - `aws ecs execute-command` debugging into the running container.
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                resources=["*"],
            )
        )

        # --- EC2 instance role for the capacity-provider ASG ----------------
        self.instance_role = iam.Role(
            self,
            "InstanceRole",
            role_name="preview-instance-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

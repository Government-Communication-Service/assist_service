"""Networking for the preview environment.

Imports the existing GCS-LLM VPC by explicit attributes (no `Vpc.from_lookup`,
no `ec2:Describe*` permissions needed at synth time). Does not modify the
existing `gcs-llm-db-sg` — it already permits tcp/5432 and tcp/443 from the
private subnets ECS tasks run in.

ALB ingress and egress-to-RDS are scoped to the same prefix list / security
group dev's ALB uses, confirmed live via AWS CLI rather than assumed.
"""

from aws_cdk import aws_ec2 as ec2
from config import Settings
from constructs import Construct


class Network(Construct):
    def __init__(self, scope: Construct, construct_id: str, settings: Settings) -> None:
        super().__init__(scope, construct_id)

        self.vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "Vpc",
            vpc_id=settings.vpc_id,
            availability_zones=settings.availability_zones,
            public_subnet_ids=settings.public_subnet_ids,
            private_subnet_ids=settings.private_subnet_ids,
        )

        # --- ALB security group: internet-facing, explicit egress only -----
        self.alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            security_group_name="preview-alb-sg",
            vpc=self.vpc,
            description="Preview ALB - public HTTP/HTTPS in, task port out",
            allow_all_outbound=False,
        )
        allowed_ips = ec2.Peer.prefix_list(settings.alb_ingress_prefix_list_id)
        self.alb_sg.add_ingress_rule(allowed_ips, ec2.Port.tcp(443), "HTTPS from the allowed-IPs prefix list")

        # --- ECS task security group: only reachable from the ALB ----------
        self.task_sg = ec2.SecurityGroup(
            self,
            "TaskSg",
            security_group_name="preview-task-sg",
            vpc=self.vpc,
            description="Preview ECS task - only reachable from the preview ALB",
            allow_all_outbound=False,
        )
        self.task_sg.add_ingress_rule(
            self.alb_sg,
            ec2.Port.tcp(settings.container_port),
            "App traffic from the preview ALB",
        )
        self.task_sg.add_egress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "AWS APIs / Bedrock / Secrets / S3 / ECR / OpenSearch"
        )
        rds_sg = ec2.SecurityGroup.from_security_group_id(self, "RdsSg", settings.rds_security_group_id, mutable=False)
        self.task_sg.add_egress_rule(rds_sg, ec2.Port.tcp(settings.rds_port), "Postgres (gcs-llm-db)")

        self.alb_sg.add_egress_rule(self.task_sg, ec2.Port.tcp(settings.container_port), "App traffic to preview tasks")

        # --- OpenSearch security group: only reachable from preview tasks --
        self.opensearch_sg = ec2.SecurityGroup(
            self,
            "OpensearchSg",
            security_group_name="preview-opensearch-sg",
            vpc=self.vpc,
            description="Preview OpenSearch domain - only reachable from preview ECS tasks",
            allow_all_outbound=False,
        )
        self.opensearch_sg.add_ingress_rule(self.task_sg, ec2.Port.tcp(443), "HTTPS from preview ECS tasks")

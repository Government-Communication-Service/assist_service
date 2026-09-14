"""ECS cluster + a single EC2 capacity provider instance.

Creates its own cluster (not the account's `default` cluster). Fixed single
instance (min=max=desired=1), ECS-managed scaling disabled.
"""

from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from config import Settings
from constructs import Construct


class Compute(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        settings: Settings,
        vpc: ec2.IVpc,
        instance_security_group: ec2.ISecurityGroup,
        instance_role: iam.IRole,
    ) -> None:
        super().__init__(scope, construct_id)

        self.cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name=settings.ecs_cluster_name,
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        self.asg = autoscaling.AutoScalingGroup(
            self,
            "Asg",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[vpc.private_subnets[0]]),
            instance_type=ec2.InstanceType(settings.instance_type),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(),
            security_group=instance_security_group,
            role=instance_role,
            min_capacity=1,
            max_capacity=1,
            desired_capacity=1,
        )

        self.capacity_provider = ecs.AsgCapacityProvider(
            self,
            "CapacityProvider",
            capacity_provider_name="preview-capacity-provider",
            auto_scaling_group=self.asg,
            enable_managed_scaling=False,
            enable_managed_termination_protection=False,
        )
        self.cluster.add_asg_capacity_provider(self.capacity_provider)

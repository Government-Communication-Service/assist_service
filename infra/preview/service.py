"""The ECS service: task definition, ALB, DNS.

Container has no `command` override — `Dockerfile.ecs` bakes the entrypoint
in as `ENTRYPOINT` (see `infra/docker-entrypoint.sh`).
"""

from aws_cdk import Duration
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_route53 as route53
from aws_cdk import aws_route53_targets as route53_targets
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_wafv2 as wafv2
from config import Settings
from constructs import Construct


class Service(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        settings: Settings,
        vpc: ec2.IVpc,
        alb_security_group: ec2.ISecurityGroup,
        task_security_group: ec2.ISecurityGroup,
        cluster: ecs.ICluster,
        capacity_provider_name: str,
        execution_role: iam.IRole,
        task_role: iam.IRole,
        repository: ecr.IRepository,
        bucket: s3.IBucket,
        opensearch_domain: opensearch.IDomain,
        image_tag: str,
        log_group: logs.ILogGroup,
    ) -> None:
        super().__init__(scope, construct_id)

        task_definition = ecs.Ec2TaskDefinition(
            self,
            "TaskDefinition",
            family="preview-copilot-api",
            network_mode=ecs.NetworkMode.AWS_VPC,
            execution_role=execution_role,
            task_role=task_role,
        )

        container = task_definition.add_container(
            "App",
            image=ecs.ContainerImage.from_ecr_repository(repository, image_tag),
            cpu=1792,
            memory_limit_mib=3584,
            logging=ecs.LogDrivers.aws_logs(stream_prefix="preview", log_group=log_group),
            environment={
                "APP_SECRET_NAME": settings.app_secret_name,
                "AWS_DEFAULT_REGION": settings.region,
                "BUGSNAG_RELEASE_STAGE": settings.bugsnag_release_stage,
                "GCS_DATA_API_URL": settings.gcs_data_api_url,
                "LITELLM_LOGGING": "True",
                "OPENSEARCH_HOST": opensearch_domain.domain_endpoint,
                "OPENSEARCH_PORT": "443",
                "PORT": str(settings.container_port),
                "POSTGRES_DB": settings.postgres_db,
                "POSTGRES_HOST": settings.rds_endpoint,
                "POSTGRES_USER": settings.postgres_user,
                "S3_ERRORDOCS_BUCKET": bucket.bucket_name,
                "URL_HOSTNAME": settings.frontend_origin,
                "USE_RAG": "True",
            },
        )
        container.add_port_mappings(ecs.PortMapping(container_port=settings.container_port, protocol=ecs.Protocol.TCP))

        service = ecs.Ec2Service(
            self,
            "Service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            security_groups=[task_security_group],
            vpc_subnets=ec2.SubnetSelection(subnets=[vpc.private_subnets[0]]),
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=capacity_provider_name, weight=1)
            ],
            enable_execute_command=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            # Startup runs a DB check + `alembic upgrade head` before gunicorn even
            # binds the port, then 3 workers each re-import the full app - without
            # this, the ALB can mark a still-starting task unhealthy and trigger the
            # circuit breaker's rollback before it's had a chance to come up.
            health_check_grace_period=Duration.seconds(120),
            # The single instance only has capacity for one task's worth of
            # cpu/memory reservation - ECS's default rolling update (start the
            # new task alongside the old one, 100-200%) can never place the
            # second task and the deployment fails with "insufficient CPU
            # units available". Stop-then-start instead: brief downtime on
            # every deploy, but it actually fits.
            min_healthy_percent=0,
            max_healthy_percent=100,
            # AZ Rebalancing requires headroom for >100% during a deployment
            # to move tasks between AZs - meaningless here anyway (single
            # instance, single AZ) and incompatible with maxHealthyPercent<=100.
            availability_zone_rebalancing=ecs.AvailabilityZoneRebalancing.DISABLED,
        )

        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self, "Zone", hosted_zone_id=settings.hosted_zone_id, zone_name=settings.hosted_zone_name
        )
        certificate = acm.Certificate(
            self, "Certificate", domain_name=settings.fqdn, validation=acm.CertificateValidation.from_dns(hosted_zone)
        )

        alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            vpc=vpc,
            internet_facing=True,
            vpc_subnets=ec2.SubnetSelection(subnets=vpc.public_subnets),
            security_group=alb_security_group,
        )

        target_group = elbv2.ApplicationTargetGroup(
            self,
            "TargetGroup",
            vpc=vpc,
            port=settings.container_port,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/healthcheck",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
            ),
        )
        service.attach_to_application_target_group(target_group)

        alb.add_listener(
            "HttpsListener",
            port=443,
            certificates=[certificate],
            default_action=elbv2.ListenerAction.forward([target_group]),
        )
        alb.add_listener(
            "HttpListener",
            port=80,
            default_action=elbv2.ListenerAction.redirect(port="443", protocol="HTTPS", permanent=True),
        )

        route53.ARecord(
            self,
            "AliasRecord",
            zone=hosted_zone,
            record_name=settings.preview_subdomain,
            target=route53.RecordTarget.from_alias(route53_targets.LoadBalancerTarget(alb)),
        )

        wafv2.CfnWebACLAssociation(
            self,
            "WafAssociation",
            resource_arn=alb.load_balancer_arn,
            web_acl_arn=settings.waf_web_acl_arn,
        )

        self.alb = alb

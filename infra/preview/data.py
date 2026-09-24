"""New preview-only data resources: S3 bucket + OpenSearch domain.

The access policy must stay `Principal: "*"` - the app authenticates with
HTTP Basic Auth, not IAM/SigV4, and unsigned requests are rejected as
"anonymous" by anything narrower. Matches dev/test/prod's own domains: the
real perimeter is VPC placement + the security group, not this policy.

Master user is `admin`, password from the generated `opensearch_password`
secret (`preview/generated_secrets.py`) - also copied into
`copilot_preview_secrets` so the app's `APP_SECRET_NAME` read matches.
"""

from aws_cdk import RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from config import Settings
from constructs import Construct


class Data(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        settings: Settings,
        vpc: ec2.IVpc,
        opensearch_sg: ec2.ISecurityGroup,
        opensearch_password: secretsmanager.ISecret,
    ) -> None:
        super().__init__(scope, construct_id)

        # --- S3 bucket -------------------------------------------------------
        self.bucket = s3.Bucket(
            self,
            "Bucket",
            bucket_name=settings.s3_bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- OpenSearch domain ------------------------------------------------
        self.domain = opensearch.Domain(
            self,
            "Domain",
            domain_name=settings.opensearch_domain_name,
            version=opensearch.EngineVersion.OPENSEARCH_2_13,
            capacity=opensearch.CapacityConfig(
                data_node_instance_type="r6g.large.search",
                data_nodes=1,
                master_nodes=0,
            ),
            ebs=opensearch.EbsOptions(volume_size=10, volume_type=ec2.EbsDeviceVolumeType.GP3),
            zone_awareness=opensearch.ZoneAwarenessConfig(enabled=False),
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnets=[vpc.private_subnets[0]])],
            security_groups=[opensearch_sg],
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            node_to_node_encryption=True,
            enforce_https=True,
            fine_grained_access_control=opensearch.AdvancedSecurityOptions(
                master_user_name="admin",
                master_user_password=opensearch_password.secret_value,
            ),
            access_policies=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AnyPrincipal()],
                    actions=["es:*"],
                    resources=["*"],
                )
            ],
            removal_policy=RemovalPolicy.DESTROY,
        )

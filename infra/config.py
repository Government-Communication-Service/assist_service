"""Settings for the preview environment.

Fields with no default are AWS identifiers for existing shared account
resources (VPC, RDS, bastion, secret, IAM policy, Route53 zone). They have no
default and are loaded from `cdk.context.json`, which is gitignored — org
policy prohibits AWS identifiers in this repo. Copy `cdk.context.json.example`
to `cdk.context.json` and fill in the real values before running `cdk
synth`/`deploy`.

Fields with a default are names for new preview-only resources — safe to
commit, since they don't identify anything that exists yet.
"""

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        json_file="cdk.context.json",
        json_file_encoding="utf-8",
        extra="ignore",
    )

    # --- existing shared VPC (GCS-LLM) --------------------------------------
    account: str
    region: str = "eu-west-2"
    vpc_id: str
    public_subnet_ids: list[str]
    private_subnet_ids: list[str]
    availability_zones: list[str]

    # --- existing shared RDS instance (hosts dev/test/prod DBs already) ----
    rds_endpoint: str
    rds_port: int = 5432
    rds_security_group_id: str  # gcs-llm-db-sg — not modified by this stack
    postgres_user: str  # preview-only role, scoped to just its own database - see infra/README.md

    # --- existing perimeter controls (shared with dev/test/prod) ------------
    alb_ingress_prefix_list_id: str  # copilot-allowed-ip-addresses
    waf_web_acl_arn: str  # dev-gcs-copilot-api-waf

    # --- existing bastion (shared DB-access bastion, not prod-specific) ----
    bastion_name_tag: str

    # --- preview's own secret (assembled from generated values, see preview/generated_secrets.py) --
    # existing IAM policy - covers any copilot_* secret name, incl. this one
    app_secret_name: str
    secrets_policy_arn: str

    # --- existing Route53 zone -----------------------------------------------
    hosted_zone_id: str
    hosted_zone_name: str

    # --- values reused verbatim from the real dev EB environment ------------
    gcs_data_api_url: str

    # --- the one branch-specific value --------------------------------------
    preview_subdomain: str = "preview"

    # CORS-allowed frontend origin. Default assumes the frontend team's
    # connect.communications.gov.uk subdomain follows {preview_subdomain}. —
    # override in cdk.context.json if their actual name differs.
    frontend_origin: str | None = None

    # --- new preview-only resource names -------------------------------------
    ecr_repo_name: str = "copilot-api"
    ecs_cluster_name: str = "assist-preview"
    opensearch_domain_name: str = "preview-assist-opensearch"  # AWS domain names are capped at 28 chars
    log_group_name: str = "/copilot/preview"
    instance_type: str = "t3.medium"

    # --- generated secrets (preview/generated_secrets.py) -------------------------------
    postgres_password_secret_name: str = "copilot-preview/postgres-password"
    auth_secret_key_secret_name: str = "copilot-preview/auth-secret-key"
    opensearch_password_secret_name: str = "copilot-preview/opensearch-password"
    container_port: int = 8000  # matches the real Procfile/gunicorn default, not the docker-compose dev port (5312)
    postgres_db: str = "preview_gcs_llm_copilot"
    bugsnag_release_stage: str = "preview"

    @model_validator(mode="after")
    def set_frontend_origin_default(self) -> "Settings":
        if self.frontend_origin is None:
            self.frontend_origin = f"https://{self.preview_subdomain}.connect.communications.gov.uk"
        return self

    @property
    def fqdn(self) -> str:
        return f"{self.preview_subdomain}.{self.hosted_zone_name}"

    @property
    def s3_bucket_name(self) -> str:
        return f"assist-preview-{self.account}-{self.region}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority order: init kwargs > env vars > cdk.context.json > defaults."""
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()

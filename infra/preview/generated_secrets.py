"""Generated values for secrets whose randomness doesn't need to come from a
human: the preview Postgres role's password, the preview Auth-Token value,
and the OpenSearch master password - plus the combined `copilot_preview_secrets`
the app actually reads, assembled from those three via CloudFormation dynamic
references (no manual copy step).

`opensearch_password` is also wired directly into the domain's master user in
`Data`. `postgres_password` still needs a one-time manual copy into the
`CREATE ROLE` statement (the DB role itself is outside CDK - see
infra/README.md). `auth_secret_key` still needs a one-time handoff to the
frontend team, since it's a shared bearer token, not something generated
independently on each side.

Characters that would break shell/SQL/JSON quoting are excluded so the
generated values can be interpolated into those contexts safely.
"""

import json

from aws_cdk import RemovalPolicy
from aws_cdk import aws_secretsmanager as secretsmanager
from config import Settings
from constructs import Construct

QUOTING_UNSAFE_CHARACTERS = "\"'@/\\`$"


def _dynamic_ref(secret_name: str) -> str:
    return f"{{{{resolve:secretsmanager:{secret_name}:SecretString}}}}"


class GeneratedSecrets(Construct):
    def __init__(self, scope: Construct, construct_id: str, settings: Settings) -> None:
        super().__init__(scope, construct_id)

        self.postgres_password = secretsmanager.Secret(
            self,
            "PostgresPassword",
            secret_name=settings.postgres_password_secret_name,
            description="Generated password for preview_gcs_llm_copilot_user - see infra/README.md",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_characters=QUOTING_UNSAFE_CHARACTERS, password_length=40
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.auth_secret_key = secretsmanager.Secret(
            self,
            "AuthSecretKey",
            secret_name=settings.auth_secret_key_secret_name,
            description="Generated Auth-Token value for preview - see infra/README.md",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_characters=QUOTING_UNSAFE_CHARACTERS, password_length=40
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.opensearch_password = secretsmanager.Secret(
            self,
            "OpensearchPassword",
            secret_name=settings.opensearch_password_secret_name,
            description="Generated OpenSearch master password, wired directly into the domain in Data",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_characters=QUOTING_UNSAFE_CHARACTERS, password_length=40
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Combined secret the app actually reads (APP_SECRET_NAME). Built from
        # dynamic references rather than GenerateSecretString, since a secret
        # can only auto-generate one random value of its own - this one instead
        # pulls the three already-generated values in by name. bugsnag_api_key
        # stays a literal placeholder until a real Bugsnag project key exists.
        self.app_secret = secretsmanager.CfnSecret(
            self,
            "CombinedSecret",
            name=settings.app_secret_name,
            description="Assembled from the generated secrets above, plus a Bugsnag placeholder - see infra/README.md",
            secret_string=json.dumps(
                {
                    "postgres_password": _dynamic_ref(settings.postgres_password_secret_name),
                    "auth_secret_key": _dynamic_ref(settings.auth_secret_key_secret_name),
                    "opensearch_password": _dynamic_ref(settings.opensearch_password_secret_name),
                    "bugsnag_api_key": "PLACEHOLDER",
                }
            ),
        )
        self.app_secret.apply_removal_policy(RemovalPolicy.DESTROY)
        # Dynamic references aren't tracked by CDK as real dependencies (they're
        # just text embedded in a string), so this needs to be explicit: without
        # it, CloudFormation could try to create CombinedSecret before the
        # secrets it reads from exist.
        self.app_secret.node.add_dependency(self.postgres_password)
        self.app_secret.node.add_dependency(self.auth_secret_key)
        self.app_secret.node.add_dependency(self.opensearch_password)

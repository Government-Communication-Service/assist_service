# ruff: noqa: E501
"""add_chat_classification_taxonomy_and_mapping_tables

Revision ID: 8834bc4acb6f
Revises: b8d3e4f1a927
Create Date: 2026-07-24 10:06:47.154306

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8834bc4acb6f"
down_revision: Union[str, None] = "b8d3e4f1a927"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEED_CLASSIFICATIONS = [
    {
        "title": "Press and media handling",
        "description": "Drafting press releases, statements, or handling responses to interview requests, journalist and media queries.",
    },
    {
        "title": "Speeches and speaking notes",
        "description": "Writing speeches, talking points, or notes for someone to deliver aloud.",
    },
    {
        "title": "Social media content and copy",
        "description": "Creating posts, captions, or copy for social media channels.",
    },
    {
        "title": "Internal communications",
        "description": "Producing internal updates, staff communications, or internal comms plans (excludes ad hoc interpersonal comms)",
    },
    {
        "title": "Campaign planning and OASIS",
        "description": "Planning a communications campaign, including structuring it around the OASIS framework.",
    },
    {
        "title": "Communications strategy development",
        "description": "Developing a broader communications strategy or approach.",
    },
    {
        "title": "Evaluation and performance",
        "description": "Defining success measures, KPIs, or evaluating communications and campaign performance.",
    },
    {
        "title": "Risk and crisis",
        "description": "Assessing communications risk or planning crisis and reactive communications.",
    },
    {
        "title": "Content quality, accessibility and proofreading",
        "description": "Checking or reviewing content quality, accessibility, editing for clarity, plain English, or proofreading against style standards, including the Gov.UK style guide.",
    },
    {
        "title": "Stakeholder engagement",
        "description": "Identifying, mapping, or planning engagement with stakeholders.",
    },
    {
        "title": "Ministerial and senior leadership communications",
        "description": "Preparing communications for ministers or senior leadership.",
    },
    {
        "title": "Summarisation and thematic analysis",
        "description": "Summarising content or identifying themes across documents or feedback.",
    },
    {
        "title": "HR, recruitment and job applications",
        "description": "Support with HR processes, recruitment, or job applications.",
    },
    {
        "title": "Events and event planning",
        "description": "Planning or organising an event.",
    },
    {
        "title": "Behavioural science and audience insight",
        "description": "Applying behavioural science or audience insight to communications: including COM-B, audience segments, and synthetic personas.",
    },
    {
        "title": "Email drafting and correspondence",
        "description": "Drafting emails or other written correspondence.",
    },
    {
        "title": "Learning, development and skills-building",
        "description": "Learning, training, or building communications skills.",
    },
    {
        "title": "Data analysis",
        "description": "Analysing data including Smart Targets, spreadsheets and advice on data analysis (except where the scope is clearly evaluation).",
    },
    {
        "title": "Searching for publicly-available information",
        "description": "Finding or verifying publicly available information.",
    },
    {
        "title": "Navigating interpersonal relationships",
        "description": "Advice on workplace relationships or interpersonal situations.",
    },
]


def upgrade() -> None:
    # 1. Create the chat_classification taxonomy table
    op.create_table(
        "chat_classification",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.dialects.postgresql.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("title", name="uq_chat_classification_title"),
    )

    # 2. Seed with the approved taxonomy (thoughts/2026-09-21.md)
    meta = sa.MetaData()
    meta.reflect(bind=op.get_bind())
    classification_table = sa.Table("chat_classification", meta)
    op.bulk_insert(classification_table, SEED_CLASSIFICATIONS)

    # 3. Create the mapping table (per-chat classification results)
    op.create_table(
        "chat_classification_mapping",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.dialects.postgresql.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("classification_id", sa.Integer(), nullable=True),
        sa.Column("topic", sa.Text(), nullable=True),
        sa.Column("discipline", sa.Text(), nullable=True),
        sa.Column("llm_internal_response_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["chat_id"], ["chat.id"]),
        sa.ForeignKeyConstraint(["classification_id"], ["chat_classification.id"]),
        sa.ForeignKeyConstraint(["llm_internal_response_id"], ["llm_internal_response.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_classification_mapping_chat_id", "chat_classification_mapping", ["chat_id"])


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chat_classification_mapping_chat_id")
    op.execute("DROP TABLE IF EXISTS chat_classification_mapping")
    op.execute("DROP TABLE IF EXISTS chat_classification")

"""
Locust load testing suite for the Copilot API.

Usage:
    cd scripts/load_testing

    # Web UI — pick profile interactively
    uv run locust -f locustfile.py --class-picker

    # CLI — run a specific profile headlessly
    uv run locust -f locustfile.py CheapSoakUser      -u 10 -r 2  -t 4h  --headless --csv=results/cheap_soak
    uv run locust -f locustfile.py DocumentStressUser  -u 8  -r 1  -t 2h  --headless --csv=results/doc_stress
    uv run locust -f locustfile.py RagLoadUser         -u 4  -r 1  -t 1h  --headless --csv=results/rag_load
    uv run locust -f locustfile.py FullMixUser         -u 12 -r 2  -t 2h  --headless --csv=results/full_mix
    uv run locust -f locustfile.py AudienceBuilderUser -u 2  -r 1  -t 1h  --headless --csv=results/ab_pipeline
    uv run locust -f locustfile.py StyleGuideUser      -u 2  -r 1  -t 1h  --headless --csv=results/style_guide

    Host defaults to BASE_URL from config.py (reads AUTH_SECRET_KEY and BASE_URL from .env.dev or .env).
    Override at runtime with --host if needed.

Profiles:
    CheapSoakUser        Zero Bedrock cost. Run for hours to detect slow memory growth.
    DocumentStressUser   Heavy document upload focus. Targets the suspected memory leak path.
    RagLoadUser          RAG-enabled chat. Tests Bedrock + OpenSearch under load.
    FullMixUser          Production-like traffic mix.
    AudienceBuilderUser  Full audience builder pipeline. Very low concurrency — high Bedrock cost per run.
    StyleGuideUser       Style guide checks via the chat endpoint. Very low concurrency — triggers Bedrock.
"""

import os
import sys

# Allow imports from this directory regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from auth import AuthMixin
from config import BASE_URL
from locust import HttpUser, between, constant, task
from tasks import audience_builder, chat, documents, lightweight, style_guide


class _BaseUser(AuthMixin, HttpUser):
    abstract = True
    host = BASE_URL


class CheapSoakUser(_BaseUser):
    """
    Zero Bedrock cost. Designed for long soak tests (4+ hours) to detect slow
    memory growth. No LLM calls — only uploads, deletes, and GET endpoints.
    Recommended: -u 10 -r 2 -t 4h
    """

    wait_time = between(3, 10)

    @task(4)
    def t_list_chats(self):
        lightweight.list_chats(self)

    @task(3)
    def t_list_documents(self):
        lightweight.list_documents(self)

    @task(3)
    def t_healthcheck(self):
        lightweight.healthcheck(self)

    @task(2)
    def t_upload_document(self):
        documents.upload_document(self)

    @task(1)
    def t_delete_document(self):
        documents.delete_document(self)


class DocumentStressUser(_BaseUser):
    """
    Heavy document upload focus. Targets the suspected memory leak in the
    unstructured library parsing path. Each upload reads the full file into
    memory and runs in a thread pool.
    Recommended: -u 8 -r 1 -t 2h
    """

    wait_time = between(3, 10)

    @task(7)
    def t_upload_document(self):
        documents.upload_document(self)

    @task(2)
    def t_list_documents(self):
        documents.list_documents(self)

    @task(1)
    def t_delete_document(self):
        documents.delete_document(self)


class RagLoadUser(_BaseUser):
    """
    RAG-enabled chat. Each request triggers index relevance check, query
    rewriting, OpenSearch searches, and chunk evaluation — all via Bedrock.
    Keep concurrency low to manage costs.
    Recommended: -u 4 -r 1 -t 1h
    """

    wait_time = between(3, 10)

    @task(5)
    def t_chat_with_rag(self):
        chat.create_chat_with_rag(self)

    @task(2)
    def t_chat_no_rag(self):
        chat.create_chat_no_rag(self)

    @task(3)
    def t_list_chats(self):
        lightweight.list_chats(self)


class FullMixUser(_BaseUser):
    """
    Realistic production-like traffic mix across all API areas.
    Recommended: -u 12 -r 2 -t 2h
    """

    wait_time = between(1, 5)

    @task(5)
    def t_upload_document(self):
        documents.upload_document(self)

    @task(2)
    def t_delete_document(self):
        documents.delete_document(self)

    @task(3)
    def t_list_documents(self):
        documents.list_documents(self)

    @task(4)
    def t_chat_with_rag(self):
        chat.create_chat_with_rag(self)

    @task(3)
    def t_chat_no_rag(self):
        chat.create_chat_no_rag(self)

    @task(2)
    def t_list_chats(self):
        lightweight.list_chats(self)

    @task(2)
    def t_style_guide_via_chat(self):
        style_guide.check_style_guide_via_chat(self)

    @task(2)
    def t_healthcheck(self):
        lightweight.healthcheck(self)


class AudienceBuilderUser(_BaseUser):
    """
    Full audience builder pipeline: upload (randomly with/without Vision processing)
    -> analyse (parallel Bedrock windows) -> citations (50% of runs) -> save.
    Every step incurs significant Bedrock cost. Keep concurrency very low.
    Analysis outputs are captured to a timestamped JSONL file in data/ignored/load_testing/.
    Recommended: -u 2 -r 1 -t 1h
    """

    wait_time = constant(180)
    fixed_count = 2

    @task
    def t_full_pipeline(self):
        audience_builder.full_pipeline(self)


class StyleGuideUser(_BaseUser):
    """
    Style guide checks via the chat endpoint (as the frontend uses it).
    Discovers the 'GOV.UK style guide checker' use case UUID on first run and
    caches it for subsequent users. Each request triggers Bedrock.
    Very low concurrency to manage costs.
    Recommended: -u 2 -r 1 -t 1h
    """

    wait_time = constant(180)
    fixed_count = 2

    @task
    def t_style_guide(self):
        style_guide.check_style_guide_via_chat(self)

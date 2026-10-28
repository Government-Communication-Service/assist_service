# ruff: noqa: E402
import logging.config
import os
import sys

from alembic import context
from alembic.config import Config
from sqlalchemy import engine_from_config, pool

logging.config.fileConfig("alembic.ini")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(repo_root)

from app.database.database_url import database_url
from app.database.models import Base


def run_migrations_online():
    print("running migrations online")

    alembic_cfg = Config("alembic.ini")
    configuration = alembic_cfg.get_section(alembic_cfg.config_ini_section)
    configuration["sqlalchemy.url"] = database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)

        with context.begin_transaction():
            context.execute("SET search_path TO public")
            context.run_migrations()


run_migrations_online()

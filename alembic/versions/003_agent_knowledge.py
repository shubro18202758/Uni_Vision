"""add agent_knowledge table for persistent learning

Revision ID: 003_agent_knowledge
Revises: 002_perf_indexes
Create Date: 2025-01-01 00:00:02.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003_agent_knowledge"
down_revision: Union[str, None] = "002_perf_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_knowledge",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("category", sa.Text(), nullable=False, server_default="general"),
        sa.Column("data", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    op.create_index(
        "idx_agent_knowledge_category",
        "agent_knowledge",
        ["category"],
    )


def downgrade() -> None:
    op.drop_index("idx_agent_knowledge_category")
    op.drop_table("agent_knowledge")

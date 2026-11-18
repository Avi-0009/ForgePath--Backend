"""Store evidence-checked, unverified project review suggestions.

Revision ID: c7af3f5d12a9
Revises: a31e8c4df209
"""

from alembic import op
import sqlalchemy as sa

from sqlalchemy.dialects.postgresql import JSONB


revision = "c7af3f5d12a9"
down_revision = "a31e8c4df209"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "project_reviews",
        sa.Column(
            "findings",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "project_reviews",
        "findings",
    )
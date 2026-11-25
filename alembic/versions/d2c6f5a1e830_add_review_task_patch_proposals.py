"""Add review-task patch proposals and decisions.

Revision ID: d2c6f5a1e830
Revises: a8d4e2c19b70
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "d2c6f5a1e830"
down_revision = "a8d4e2c19b70"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "project_review_tasks",
        sa.Column(
            "patch_status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'not_started'"),
        ),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("patch_token", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("patch_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("patch", JSONB(), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("patch_error", sa.String(length=300), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("patched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        "patch_status IN ('not_started', 'generating', 'ready', 'approved', 'rejected', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        type_="check",
    )
    op.drop_column("project_review_tasks", "patched_at")
    op.drop_column("project_review_tasks", "patch_error")
    op.drop_column("project_review_tasks", "patch")
    op.drop_column("project_review_tasks", "patch_started_at")
    op.drop_column("project_review_tasks", "patch_token")
    op.drop_column("project_review_tasks", "patch_status")

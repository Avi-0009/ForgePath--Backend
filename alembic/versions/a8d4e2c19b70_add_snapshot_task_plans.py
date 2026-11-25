"""Add snapshot-pinned plans to review tasks.

Revision ID: a8d4e2c19b70
Revises: f734bc81a902
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "a8d4e2c19b70"
down_revision = "f734bc81a902"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "project_review_tasks",
        sa.Column("snapshot_id", UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE project_review_tasks AS task
        SET snapshot_id = review.snapshot_id
        FROM project_reviews AS review
        WHERE review.id = task.review_id
        """
    )
    op.alter_column(
        "project_review_tasks",
        "snapshot_id",
        nullable=False,
    )
    op.add_column(
        "project_review_tasks",
        sa.Column(
            "planning_status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'not_started'"),
        ),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("planning_token", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("planning_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("plan", JSONB(), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("planning_error", sa.String(length=300), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("planned_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_project_review_tasks_planning_status",
        "project_review_tasks",
        "planning_status IN ('not_started', 'planning', 'ready', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_project_review_tasks_planning_status",
        "project_review_tasks",
        type_="check",
    )
    op.drop_column("project_review_tasks", "planned_at")
    op.drop_column("project_review_tasks", "planning_error")
    op.drop_column("project_review_tasks", "plan")
    op.drop_column("project_review_tasks", "planning_started_at")
    op.drop_column("project_review_tasks", "planning_token")
    op.drop_column("project_review_tasks", "planning_status")
    op.drop_column("project_review_tasks", "snapshot_id")

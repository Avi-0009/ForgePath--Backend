"""Track approved patches detected in later snapshots.

Revision ID: e91b7a4c2d60
Revises: d2c6f5a1e830
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "e91b7a4c2d60"
down_revision = "d2c6f5a1e830"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        "patch_status IN ('not_started', 'generating', 'ready', 'approved', 'rejected', 'failed', 'applied')",
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("applied_snapshot_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE project_review_tasks
        SET patch_status = 'approved', applied_snapshot_id = NULL, applied_at = NULL
        WHERE patch_status = 'applied'
        """
    )
    op.drop_column("project_review_tasks", "applied_at")
    op.drop_column("project_review_tasks", "applied_snapshot_id")
    op.drop_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_project_review_tasks_patch_status",
        "project_review_tasks",
        "patch_status IN ('not_started', 'generating', 'ready', 'approved', 'rejected', 'failed')",
    )

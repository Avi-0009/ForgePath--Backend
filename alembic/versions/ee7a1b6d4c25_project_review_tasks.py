"""Add user-owned review finding tasks.

Revision ID: ee7a1b6d4c25
Revises: c7af3f5d12a9
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "ee7a1b6d4c25"
down_revision = "c7af3f5d12a9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_review_tasks",

        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column(
            "review_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "project_reviews.id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),

        sa.Column(
            "finding_index",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "title",
            sa.String(length=160),
            nullable=False,
        ),

        sa.Column(
            "path",
            sa.String(length=1024),
            nullable=False,
        ),

        sa.Column(
            "severity",
            sa.String(length=12),
            nullable=False,
        ),

        sa.Column(
            "evidence",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "explanation",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'proposed'"),
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),

        sa.UniqueConstraint(
            "review_id",
            "finding_index",
            name="uq_project_review_tasks_finding",
        ),

        sa.CheckConstraint(
            "finding_index >= 0",
            name="ck_project_review_tasks_finding_index",
        ),

        sa.CheckConstraint(
            "status IN ('proposed', 'approved', 'dismissed')",
            name="ck_project_review_tasks_status",
        ),

        sa.CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low', 'info')",
            name="ck_project_review_tasks_severity",
        ),
    )

    op.create_index(
        "ix_project_review_tasks_owner_project_created",
        "project_review_tasks",
        ["user_id", "project_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_project_review_tasks_owner_project_created",
        table_name="project_review_tasks",
    )

    op.drop_table("project_review_tasks")
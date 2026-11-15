"""Add bounded snapshot review job history.

Revision ID: a31e8c4df209
Revises: f62e7b9c4a10
"""
from alembic import op
import sqlalchemy as sa

revision = "a31e8c4df209"
down_revision = "f62e7b9c4a10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_reviews",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.UUID(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("snapshot_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("files_scanned", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("files_reviewed", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("ai_calls", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("result_text", sa.Text(), nullable=True),
        sa.Column("error_message", sa.String(300), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'partial', 'failed')", name="ck_project_reviews_status"),
    )
    op.create_index("ix_project_reviews_user_id", "project_reviews", ["user_id"])
    op.create_index("ix_project_reviews_project_id", "project_reviews", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_project_reviews_project_id", table_name="project_reviews")
    op.drop_index("ix_project_reviews_user_id", table_name="project_reviews")
    op.drop_table("project_reviews")
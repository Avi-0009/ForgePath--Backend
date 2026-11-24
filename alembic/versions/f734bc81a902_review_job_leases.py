"""Add fenced review leases and interrupted state.

Revision ID: f734bc81a902
Revises: ee7a1b6d4c25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "f734bc81a902"
down_revision = "ee7a1b6d4c25"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("project_reviews", sa.Column("lease_token", UUID(as_uuid=True), nullable=True))
    op.add_column("project_reviews", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.drop_constraint("ck_project_reviews_status", "project_reviews", type_="check")
    op.create_check_constraint("ck_project_reviews_status", "project_reviews",
                               "status IN ('queued','running','completed','partial','failed','interrupted')")
    # Stop old workers before deployment. Old attempts cannot be safely adopted.
    op.execute("UPDATE project_reviews SET status='interrupted', updated_at=now(), error_message='Review interrupted during upgrade. Confirm a new review to retry.' WHERE status IN ('queued','running')")
    op.create_index("uq_project_reviews_active_user", "project_reviews", ["user_id"], unique=True,
                    postgresql_where=sa.text("status IN ('queued','running')"))


def downgrade():
    op.drop_index("uq_project_reviews_active_user", table_name="project_reviews")
    op.execute("UPDATE project_reviews SET status='failed' WHERE status='interrupted'")
    op.drop_constraint("ck_project_reviews_status", "project_reviews", type_="check")
    op.create_check_constraint("ck_project_reviews_status", "project_reviews",
                               "status IN ('queued','running','completed','partial','failed')")
    op.drop_column("project_reviews", "lease_expires_at")
    op.drop_column("project_reviews", "lease_token")

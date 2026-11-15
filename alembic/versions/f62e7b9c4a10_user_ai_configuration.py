"""Store user-scoped encrypted AI credentials.

Revision ID: f62e7b9c4a10
Revises: 948b0e1cc2a6
"""
from alembic import op
import sqlalchemy as sa

revision = "f62e7b9c4a10"
down_revision = "948b0e1cc2a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_ai_configurations",
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model_identifier", sa.String(255), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", name="uq_user_ai_configurations_user"),
        sa.CheckConstraint("provider IN ('nvidia', 'openai', 'huggingface')", name="ck_user_ai_provider"),
    )


def downgrade() -> None:
    op.drop_table("user_ai_configurations")
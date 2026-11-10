"""create agent workflow history

Revision ID: 689465768966
Revises: 2c0aae3a2185
Create Date: 2026-11-09 22:25:08.931137

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "689465768966"
down_revision: Union[str, Sequence[str], None] = "2c0aae3a2185"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_workflows",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("workspace_root", sa.String(1024), nullable=False),
        sa.Column("task_description", sa.Text(), nullable=False),
        sa.Column("planning_mode", sa.String(32), nullable=False),
        sa.Column("test_runner", sa.String(32), nullable=False),
        sa.Column(
            "status",
            sa.String(64),
            server_default=sa.text("'running'"),
            nullable=False,
        ),
        sa.Column("message", sa.String(2000), nullable=True),
        sa.Column(
            "total_steps",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "completed_steps",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "repair_attempts",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "completed_steps >= 0",
            name="ck_agent_workflows_completed_steps",
        ),
        sa.CheckConstraint(
            "repair_attempts >= 0",
            name="ck_agent_workflows_repair_attempts",
        ),
        sa.CheckConstraint(
            "total_steps >= 0",
            name="ck_agent_workflows_total_steps",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "thread_id",
            name="uq_agent_workflows_thread_id",
        ),
    )

    op.create_index(
        "ix_agent_workflows_user_created_at",
        "agent_workflows",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_workflows_user_created_at",
        table_name="agent_workflows",
    )

    op.drop_table("agent_workflows")
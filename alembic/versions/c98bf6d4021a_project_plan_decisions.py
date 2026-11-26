"""Require a human decision on a saved project implementation plan.

Revision ID: c98bf6d4021a
Revises: e91b7a4c2d60
"""
from alembic import op
import sqlalchemy as sa

revision = "c98bf6d4021a"
down_revision = "e91b7a4c2d60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "project_review_tasks",
        sa.Column("plan_decision", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "project_review_tasks",
        sa.Column("plan_decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_project_review_tasks_plan_decision",
        "project_review_tasks",
        "plan_decision IN ('pending','approved','rejected')",
    )

    # Existing plans that already produced patches remain usable.
    op.execute("""
        UPDATE project_review_tasks
        SET plan_decision = 'approved',
            plan_decided_at = patched_at
        WHERE jsonb_typeof(plan) = 'object'
          AND jsonb_typeof(patch) = 'object'
    """)

    # Other historical plans now wait at the new human checkpoint.
    op.execute("""
        UPDATE project_review_tasks
        SET plan_decision = 'pending'
        WHERE jsonb_typeof(plan) = 'object'
          AND plan_decision IS NULL
    """)


def downgrade() -> None:
    op.drop_constraint(
        "ck_project_review_tasks_plan_decision",
        "project_review_tasks",
        type_="check",
    )
    op.drop_column("project_review_tasks", "plan_decided_at")
    op.drop_column("project_review_tasks", "plan_decision")

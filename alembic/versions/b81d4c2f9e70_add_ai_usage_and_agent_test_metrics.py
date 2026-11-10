"""add AI usage events and agent test-run metrics

Revision ID: b81d4c2f9e70
Revises: 689465768966
Create Date: 2026-11-10

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b81d4c2f9e70"
down_revision: Union[str, Sequence[str], None] = "689465768966"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Extend the existing workflow history row with aggregate metrics.
    workflow_metric_columns = (
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_cached_input_tokens",
        "total_reasoning_tokens",
        "ai_request_count",
        "failed_ai_request_count",
        "total_ai_duration_ms",
        "test_run_count",
        "test_pass_count",
        "test_fail_count",
        "test_timeout_count",
        "test_error_count",
        "total_test_duration_ms",
    )

    for column_name in workflow_metric_columns:
        op.add_column(
            "agent_workflows",
            sa.Column(
                column_name,
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            ),
        )

    op.create_check_constraint(
        "ck_agent_workflows_total_input_tokens",
        "agent_workflows",
        "total_input_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_output_tokens",
        "agent_workflows",
        "total_output_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_tokens",
        "agent_workflows",
        "total_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_cached_input_tokens",
        "agent_workflows",
        "total_cached_input_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_reasoning_tokens",
        "agent_workflows",
        "total_reasoning_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_ai_request_counts",
        "agent_workflows",
        "ai_request_count >= 0 AND failed_ai_request_count >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_ai_duration_ms",
        "agent_workflows",
        "total_ai_duration_ms >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_test_run_counts",
        "agent_workflows",
        "test_run_count >= 0 AND test_pass_count >= 0 "
        "AND test_fail_count >= 0 AND test_timeout_count >= 0 "
        "AND test_error_count >= 0",
    )
    op.create_check_constraint(
        "ck_agent_workflows_total_test_duration_ms",
        "agent_workflows",
        "total_test_duration_ms >= 0",
    )

    # One row per AI request. Prompts, source code, tokens/secrets, and raw
    # provider error messages are intentionally not stored in this table.
    op.create_table(
        "ai_usage_events",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("thread_id", sa.UUID(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("api_model", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column(
            "request_type",
            sa.String(length=32),
            server_default=sa.text("'chat_completion'"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "total_tokens_calculated",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed')",
            name="ck_ai_usage_events_status",
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_ai_usage_events_input_tokens",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_ai_usage_events_output_tokens",
        ),
        sa.CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_ai_usage_events_total_tokens",
        ),
        sa.CheckConstraint(
            "cached_input_tokens IS NULL OR cached_input_tokens >= 0",
            name="ck_ai_usage_events_cached_input_tokens",
        ),
        sa.CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name="ck_ai_usage_events_reasoning_tokens",
        ),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_ai_usage_events_duration_ms",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_workflows.thread_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_ai_usage_events_user_created_at",
        "ai_usage_events",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_ai_usage_events_thread_created_at",
        "ai_usage_events",
        ["thread_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_ai_usage_events_provider_model_created_at",
        "ai_usage_events",
        ["provider", "api_model", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_ai_usage_events_operation_created_at",
        "ai_usage_events",
        ["operation", "created_at"],
        unique=False,
    )

    # One row per local test execution. Do not persist stdout, stderr,
    # workspace paths, or the shell command itself.
    op.create_table(
        "agent_test_runs",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("execution_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("thread_id", sa.UUID(), nullable=True),
        sa.Column("runner", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column(
            "stdout_truncated",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "stderr_truncated",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "runner IN ('pytest', 'npm', 'go')",
            name="ck_agent_test_runs_runner",
        ),
        sa.CheckConstraint(
            "status IN ('passed', 'failed', 'timed_out', 'error')",
            name="ck_agent_test_runs_status",
        ),
        sa.CheckConstraint(
            "duration_ms >= 0",
            name="ck_agent_test_runs_duration_ms",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_workflows.thread_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "execution_id",
            name="uq_agent_test_runs_execution_id",
        ),
    )

    op.create_index(
        "ix_agent_test_runs_user_created_at",
        "agent_test_runs",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_test_runs_thread_created_at",
        "agent_test_runs",
        ["thread_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_test_runs_status_created_at",
        "agent_test_runs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_test_runs_status_created_at",
        table_name="agent_test_runs",
    )
    op.drop_index(
        "ix_agent_test_runs_thread_created_at",
        table_name="agent_test_runs",
    )
    op.drop_index(
        "ix_agent_test_runs_user_created_at",
        table_name="agent_test_runs",
    )
    op.drop_table("agent_test_runs")

    op.drop_index(
        "ix_ai_usage_events_operation_created_at",
        table_name="ai_usage_events",
    )
    op.drop_index(
        "ix_ai_usage_events_provider_model_created_at",
        table_name="ai_usage_events",
    )
    op.drop_index(
        "ix_ai_usage_events_thread_created_at",
        table_name="ai_usage_events",
    )
    op.drop_index(
        "ix_ai_usage_events_user_created_at",
        table_name="ai_usage_events",
    )
    op.drop_table("ai_usage_events")

    workflow_constraints = (
        "ck_agent_workflows_total_test_duration_ms",
        "ck_agent_workflows_test_run_counts",
        "ck_agent_workflows_total_ai_duration_ms",
        "ck_agent_workflows_ai_request_counts",
        "ck_agent_workflows_total_reasoning_tokens",
        "ck_agent_workflows_total_cached_input_tokens",
        "ck_agent_workflows_total_tokens",
        "ck_agent_workflows_total_output_tokens",
        "ck_agent_workflows_total_input_tokens",
    )

    for constraint_name in workflow_constraints:
        op.drop_constraint(
            constraint_name,
            "agent_workflows",
            type_="check",
        )

    workflow_metric_columns = (
        "total_test_duration_ms",
        "test_error_count",
        "test_timeout_count",
        "test_fail_count",
        "test_pass_count",
        "test_run_count",
        "total_ai_duration_ms",
        "failed_ai_request_count",
        "ai_request_count",
        "total_reasoning_tokens",
        "total_cached_input_tokens",
        "total_tokens",
        "total_output_tokens",
        "total_input_tokens",
    )

    for column_name in workflow_metric_columns:
        op.drop_column("agent_workflows", column_name)
from app.models import AgentTestRun, AgentWorkflowHistory, AIUsageEvent


def test_milestone_8_models_are_registered_in_metadata():
    assert AgentWorkflowHistory.__tablename__ == "agent_workflows"
    assert AIUsageEvent.__tablename__ == "ai_usage_events"
    assert AgentTestRun.__tablename__ == "agent_test_runs"


def test_ai_usage_event_keeps_optional_usage_values_nullable():
    columns = AIUsageEvent.__table__.c

    assert columns.input_tokens.nullable is True
    assert columns.output_tokens.nullable is True
    assert columns.total_tokens.nullable is True
    assert columns.cached_input_tokens.nullable is True
    assert columns.reasoning_tokens.nullable is True


def test_ai_usage_event_does_not_store_prompt_or_response_content():
    columns = set(AIUsageEvent.__table__.c.keys())

    assert "prompt" not in columns
    assert "instructions" not in columns
    assert "source_code" not in columns
    assert "response_text" not in columns
    assert "api_key" not in columns


def test_agent_test_run_does_not_store_raw_test_output_or_workspace_path():
    columns = set(AgentTestRun.__table__.c.keys())

    assert "stdout" not in columns
    assert "stderr" not in columns
    assert "command" not in columns
    assert "workspace_root" not in columns


def test_agent_workflow_history_has_usage_and_execution_aggregates():
    columns = set(AgentWorkflowHistory.__table__.c.keys())

    expected = {
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
    }

    assert expected.issubset(columns)
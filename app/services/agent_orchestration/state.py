from typing import Any, Literal, NotRequired, TypedDict


WorkflowStatus = Literal[
    "running",
    "awaiting_plan_approval",
    "awaiting_file_selection",
    "awaiting_patch_approval",
    "awaiting_test_approval",
    "completed",
    "completed_without_tests",
    "tests_failed",
    "cancelled",
    "failed",
    "awaiting_repair_file_selection",
]


class AgentWorkflowState(TypedDict):
    """Shared state passed between LangGraph workflow nodes."""

    # Workflow identity and request
    thread_id: str
    owner_user_id: str
    workspace_root: str
    task_description: str
    planning_mode: str
    additional_context: str | None
    max_steps: int

    # Test execution configuration
    test_runner: str
    test_timeout_seconds: int

    # Workflow progress
    status: NotRequired[WorkflowStatus]
    message: NotRequired[str]
    error_message: NotRequired[str | None]

    # Generated implementation plan
    plan: NotRequired[dict[str, Any] | None]
    current_step_index: NotRequired[int]

    # Current step's selected source file
    selected_file: NotRequired[str | None]

    # Pending human approval or selection
    pending_action: NotRequired[dict[str, Any] | None]

    # Current proposed code change
    patch_proposal: NotRequired[dict[str, Any] | None]

    # History of successfully applied changes
    applied_patches: NotRequired[list[dict[str, Any]]]

    completed_steps: NotRequired[list[int]]

    # Steps that require manual handling
    manual_steps: NotRequired[list[int]]
    skipped_steps: NotRequired[list[int]]

    # Latest test execution result
    test_result: NotRequired[dict[str, Any] | None]

    # Test failure analysis and bounded recovery
    failure_analysis: NotRequired[dict[str, Any] | None]
    repair_mode: NotRequired[bool]
    repair_attempts: NotRequired[int]
    max_repair_attempts: NotRequired[int]
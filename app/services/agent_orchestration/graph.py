from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.services.agent_orchestration.state import AgentWorkflowState
from app.services.agent_orchestration.nodes import (
    analyze_test_failure_node,
    approve_and_apply_patch_node,
    approve_and_run_tests_node,
    approve_plan_node,
    create_plan_node,
    generate_patch_node,
    generate_repair_patch_node,
    prepare_step_node,
    select_file_node,
    select_repair_file_node,
)


def route_after_plan_approval(
    state: AgentWorkflowState,
) -> Literal["prepare_step", "__end__"]:
    if state.get("status") == "running":
        return "prepare_step"

    return END


def route_after_step_preparation(
    state: AgentWorkflowState,
) -> Literal[
    "prepare_step",
    "select_file",
    "approve_and_run_tests",
    "__end__",
]:
    status = state.get("status")

    if status == "awaiting_file_selection":
        return "select_file"

    if status == "awaiting_test_approval":
        return "approve_and_run_tests"

    if status == "running":
        return "prepare_step"

    return END


def route_after_file_selection(
    state: AgentWorkflowState,
) -> Literal["generate_patch", "__end__"]:
    if (
        state.get("status") == "running"
        and state.get("selected_file")
    ):
        return "generate_patch"

    return END


def route_after_patch_generation(
    state: AgentWorkflowState,
) -> Literal["approve_and_apply_patch", "__end__"]:
    if state.get("status") == "awaiting_patch_approval":
        return "approve_and_apply_patch"

    return END


def route_after_patch_approval(
    state: AgentWorkflowState,
) -> Literal[
    "prepare_step",
    "approve_and_run_tests",
    "__end__",
]:
    status = state.get("status")

    # Repair patch applied: request approval to rerun tests.
    if (
        status == "awaiting_test_approval"
        and state.get("repair_mode")
    ):
        return "approve_and_run_tests"

    # Normal implementation patch applied or rejected.
    if status == "running":
        return "prepare_step"

    return END


def route_after_plan_creation(
    state: AgentWorkflowState,
) -> Literal["approve_plan", "__end__"]:
    if state.get("status") == "awaiting_plan_approval":
        return "approve_plan"

    return END


def route_after_test_execution(
    state: AgentWorkflowState,
) -> Literal["analyze_test_failure", "__end__"]:
    """Analyze completed test runs that did not pass."""

    test_result = state.get("test_result")

    if (
        state.get("status") == "running"
        and isinstance(test_result, dict)
        and test_result.get("status") != "passed"
    ):
        return "analyze_test_failure"

    return END


def route_after_failure_analysis(
    state: AgentWorkflowState,
) -> Literal["select_repair_file", "__end__"]:
    if state.get("status") == "awaiting_repair_file_selection":
        return "select_repair_file"

    return END


def route_after_repair_file_selection(
    state: AgentWorkflowState,
) -> Literal["generate_repair_patch", "__end__"]:
    if (
        state.get("status") == "running"
        and state.get("selected_file")
    ):
        return "generate_repair_patch"

    return END


def build_agent_graph(checkpointer):
    """
    Build the resumable ForgePath agent workflow.

    A durable checkpointer is required because the graph
    contains human approval and file-selection interrupts.
    """

    if checkpointer is None:
        raise ValueError(
            "A checkpointer is required for agent orchestration."
        )

    builder = StateGraph(AgentWorkflowState)

    # Existing workflow nodes.
    builder.add_node("create_plan", create_plan_node)
    builder.add_node("approve_plan", approve_plan_node)
    builder.add_node("prepare_step", prepare_step_node)
    builder.add_node("select_file", select_file_node)
    builder.add_node("generate_patch", generate_patch_node)
    builder.add_node(
        "approve_and_apply_patch",
        approve_and_apply_patch_node,
    )
    builder.add_node(
        "approve_and_run_tests",
        approve_and_run_tests_node,
    )

    # Test-failure repair nodes.
    builder.add_node(
        "analyze_test_failure",
        analyze_test_failure_node,
    )
    builder.add_node(
        "select_repair_file",
        select_repair_file_node,
    )
    builder.add_node(
        "generate_repair_patch",
        generate_repair_patch_node,
    )

    # Workflow entry.
    builder.add_edge(START, "create_plan")

    # Planning and approval.
    builder.add_conditional_edges(
        "create_plan",
        route_after_plan_creation,
    )
    builder.add_conditional_edges(
        "approve_plan",
        route_after_plan_approval,
    )

    # Normal implementation loop.
    builder.add_conditional_edges(
        "prepare_step",
        route_after_step_preparation,
    )
    builder.add_conditional_edges(
        "select_file",
        route_after_file_selection,
    )
    builder.add_conditional_edges(
        "generate_patch",
        route_after_patch_generation,
    )
    builder.add_conditional_edges(
        "approve_and_apply_patch",
        route_after_patch_approval,
    )

    # Repair workflow.
    builder.add_conditional_edges(
        "approve_and_run_tests",
        route_after_test_execution,
    )
    builder.add_conditional_edges(
        "analyze_test_failure",
        route_after_failure_analysis,
    )
    builder.add_conditional_edges(
        "select_repair_file",
        route_after_repair_file_selection,
    )
    builder.add_conditional_edges(
        "generate_repair_patch",
        route_after_patch_generation,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name="forgepath_agent",
    )
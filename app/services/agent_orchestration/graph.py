from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.services.agent_orchestration.state import AgentWorkflowState
from app.services.agent_orchestration.nodes import (
    approve_and_apply_patch_node,
    approve_and_run_tests_node,
    approve_plan_node,
    create_plan_node,
    generate_patch_node,
    prepare_step_node,
    select_file_node,
)


def route_after_plan_approval(
    state: AgentWorkflowState,
) -> Literal["prepare_step", "__end__"]:
    """Continue only after the user approves the plan."""

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
    """Choose the next action for the current implementation step."""

    status = state.get("status")

    if status == "awaiting_file_selection":
        return "select_file"

    if status == "awaiting_test_approval":
        return "approve_and_run_tests"

    if status == "running":
        # A step may have been marked for manual handling.
        # Continue to inspect the following step.
        return "prepare_step"

    return END


def route_after_file_selection(
    state: AgentWorkflowState,
) -> Literal["generate_patch", "__end__"]:
    """Generate a patch only after a valid file selection."""

    if (
        state.get("status") == "running"
        and state.get("selected_file")
    ):
        return "generate_patch"

    return END


def route_after_patch_generation(
    state: AgentWorkflowState,
) -> Literal["approve_and_apply_patch", "__end__"]:
    """Require approval before any patch can be applied."""

    if state.get("status") == "awaiting_patch_approval":
        return "approve_and_apply_patch"

    return END


def route_after_patch_approval(
    state: AgentWorkflowState,
) -> Literal["prepare_step", "__end__"]:
    """Continue only after the patch node has completed its decision."""

    if state.get("status") == "running":
        return "prepare_step"

    return END


def route_after_plan_creation(
    state: AgentWorkflowState,
) -> Literal["approve_plan", "__end__"]:
    """Proceed to approval only when a valid plan was generated."""

    if state.get("status") == "awaiting_plan_approval":
        return "approve_plan"

    return END


def build_agent_graph(checkpointer):
    """
    Build the resumable ForgePath agent workflow.

    A durable checkpointer is required because the graph contains
    human approval interrupts.
    """

    if checkpointer is None:
        raise ValueError(
            "A checkpointer is required for agent orchestration."
        )

    builder = StateGraph(AgentWorkflowState)

    # Register workflow nodes.
    builder.add_node(
        "create_plan",
        create_plan_node,
    )

    builder.add_node(
        "approve_plan",
        approve_plan_node,
    )

    builder.add_node(
        "prepare_step",
        prepare_step_node,
    )

    builder.add_node(
        "select_file",
        select_file_node,
    )

    builder.add_node(
        "generate_patch",
        generate_patch_node,
    )

    builder.add_node(
        "approve_and_apply_patch",
        approve_and_apply_patch_node,
    )

    builder.add_node(
        "approve_and_run_tests",
        approve_and_run_tests_node,
    )

    # Workflow entry.
    builder.add_edge(
        START,
        "create_plan",
    )

    # Planning and approval.
    builder.add_conditional_edges(
        "create_plan",
        route_after_plan_creation,
    )

    builder.add_conditional_edges(
        "approve_plan",
        route_after_plan_approval,
    )

    # Implementation step loop.
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

    # Final testing checkpoint.
    builder.add_edge(
        "approve_and_run_tests",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name="forgepath_agent",
    )
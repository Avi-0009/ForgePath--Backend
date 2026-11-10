import asyncio
import json
import logging
from typing import Any
from dataclasses import asdict
from langgraph.types import interrupt

from app.core.ai_provider import AIProviderError
from app.schemas.agent_orchestration import (
    AgentWorkflowResumeRequest,
)
from app.schemas.code_patch import (
    CodePatchApplyRequest,
    CodePatchRequest,
    CodePatchResponse,
)
from app.schemas.execution import TestExecutionResponse
from app.schemas.task_planning import (
    TaskPlanningRequest,
    TaskPlanningResponse,
)
from app.services.code_patch import (
    CodePatchError,
    generate_code_patch,
)
from app.services.code_patch_apply import (
    PatchApplyError,
    apply_code_patch,
)
from app.services.local_context import (
    LocalContextError,
    get_project_instructions,
)
from app.services.task_planning.planner import (
    TaskPlanningError,
    generate_task_plan,
)
from app.services.test_execution.runner import (
    TestExecutionError,
    execute_tests,
)
from app.services.workspace import (
    WorkspaceReadError,
    read_configured_workspace_file,
)
from app.services.agent_orchestration.state import AgentWorkflowState
from app.schemas.error_analysis import TestFailureAnalysis
from app.services.agent_orchestration.error_analysis import (
    TestFailureAnalysisError,
    analyze_test_failure,
    sanitize_test_output,
)
from app.services.ai_usage_tracking import record_agent_test_run



logger = logging.getLogger(__name__)


def _failure(message: str, error: str | None = None) -> dict:
    """Return a consistent terminal workflow failure."""

    return {
        "status": "failed",
        "message": message,
        "error_message": error or message,
        "pending_action": None,
    }


def _add_unique(values: list[int], value: int) -> list[int]:
    """Append an integer without creating duplicates."""

    if value not in values:
        values.append(value)

    return values


def _current_step(state: AgentWorkflowState) -> dict[str, Any]:
    """Retrieve the current plan step safely."""

    plan = TaskPlanningResponse.model_validate(
        state["plan"]
    )

    index = state.get("current_step_index", 0)

    if index < 0 or index >= len(plan.steps):
        raise ValueError("The current workflow step is invalid.")

    return plan.steps[index].model_dump(mode="json")


async def create_plan_node(
    state: AgentWorkflowState,
) -> dict:
    """Generate the implementation plan for this workflow."""

    try:
        project_instructions = None

        try:
            context = await get_project_instructions(
                project_root=state["workspace_root"],
            )

            if context is not None:
                project_instructions = context.instructions

        except LocalContextError:
            logger.warning(
                "Could not load saved instructions for agent planning."
            )

        request = TaskPlanningRequest(
            workspace_root=state["workspace_root"],
            task_description=state["task_description"],
            planning_mode=state["planning_mode"],
            additional_context=state.get("additional_context"),
            max_steps=state["max_steps"],
        )

        plan = await generate_task_plan(
            request,
            project_instructions=project_instructions,
        )

        return {
            "plan": plan.model_dump(mode="json"),
            "current_step_index": 0,
            "completed_steps": [],
            "manual_steps": [],
            "skipped_steps": [],
            "applied_patches": [],
            "pending_action": {
                "kind": "plan_approval",
                "message": "Review and approve the implementation plan.",
            },
            "status": "awaiting_plan_approval",
            "message": "Implementation plan generated. Approval required.",
            "error_message": None,
        }

    except (
        AIProviderError,
        TaskPlanningError,
        WorkspaceReadError,
    ) as exc:
        logger.warning("Agent planning failed: %s", exc)

        return _failure(
            "Could not generate the implementation plan.",
            str(exc),
        )

    except Exception:
        logger.exception("Unexpected error while creating agent plan.")

        return _failure(
            "Unexpected error while generating the implementation plan."
        )


async def approve_plan_node(
    state: AgentWorkflowState,
) -> dict:
    """Pause until the user approves or rejects the plan."""

    decision = interrupt(
        {
            "kind": "plan_approval",
            "message": "Do you approve this implementation plan?",
            "plan": state["plan"],
        },
        response_schema=AgentWorkflowResumeRequest,
    )

    if decision.action == "cancel":
        return {
            "status": "cancelled",
            "message": "Workflow cancelled by the user.",
            "pending_action": None,
        }

    if decision.action == "reject":
        return {
            "status": "cancelled",
            "message": "The implementation plan was rejected.",
            "pending_action": None,
        }

    if decision.action != "approve":
        return _failure(
            "Invalid decision for plan approval."
        )

    return {
        "status": "running",
        "message": "Plan approved. Preparing the first implementation step.",
        "current_step_index": 0,
        "pending_action": None,
    }


def prepare_step_node(
    state: AgentWorkflowState,
) -> dict:
    """Select the next executable plan step."""

    try:
        plan = TaskPlanningResponse.model_validate(
            state["plan"]
        )

        index = state.get("current_step_index", 0)

        if index >= len(plan.steps):
            return {
                "status": "awaiting_test_approval",
                "pending_action": {
                    "kind": "test_approval",
                    "message": (
                        "Review the completed implementation steps "
                        "and approve test execution."
                    ),
                },
                "selected_file": None,
                "message": "All planned steps have been processed.",
            }

        step = plan.steps[index]
        step_number = step.step_number

        completed = list(
            state.get("completed_steps", [])
        )

        manual_steps = list(
            state.get("manual_steps", [])
        )

        missing_dependencies = [
            dependency
            for dependency in step.dependencies
            if dependency not in completed
        ]

        if missing_dependencies:
            _add_unique(manual_steps, step_number)

            return {
                "current_step_index": index + 1,
                "manual_steps": manual_steps,
                "status": "running",
                "message": (
                    f"Step {step_number} requires unfinished dependencies. "
                    "It has been marked for manual handling."
                ),
            }

        available_files = list(
            step.candidate_files_to_inspect
        )

        if not available_files:
            _add_unique(manual_steps, step_number)

            return {
                "current_step_index": index + 1,
                "manual_steps": manual_steps,
                "status": "running",
                "message": (
                    f"Step {step_number} has no verified candidate files "
                    "and requires manual handling."
                ),
            }

        return {
            "status": "awaiting_file_selection",
            "selected_file": None,
            "pending_action": {
                "kind": "file_selection",
                "message": (
                    f"Select a file for step {step_number}: {step.title}"
                ),
                "step_number": step_number,
                "available_files": available_files,
            },
            "message": f"Step {step_number} is ready for file selection.",
        }

    except Exception as exc:
        logger.exception("Could not prepare the next agent step.")

        return _failure(
            "Could not prepare the next implementation step.",
            str(exc),
        )


def select_file_node(
    state: AgentWorkflowState,
) -> dict:
    """Pause until the user selects a candidate source file."""

    pending = state.get("pending_action") or {}

    decision = interrupt(
        {
            "kind": "file_selection",
            "message": pending.get(
                "message",
                "Select a source file for this step.",
            ),
            "step_number": pending.get("step_number"),
            "available_files": pending.get("available_files", []),
        },
        response_schema=AgentWorkflowResumeRequest,
    )

    if decision.action == "cancel":
        return {
            "status": "cancelled",
            "message": "Workflow cancelled during file selection.",
            "pending_action": None,
        }

    if decision.action != "select_file":
        return _failure(
            "A valid file selection is required."
        )

    available_files = pending.get("available_files", [])

    if decision.selected_file not in available_files:
        return _failure(
            "The selected file is not one of the available project files."
        )

    return {
        "selected_file": decision.selected_file,
        "status": "running",
        "pending_action": None,
        "message": (
            f"Selected {decision.selected_file}. "
            "Generating a patch proposal."
        ),
    }


async def generate_patch_node(
    state: AgentWorkflowState,
) -> dict:
    """Generate a patch proposal for the currently selected file."""

    try:
        selected_file = state.get("selected_file")

        if not selected_file:
            raise ValueError("No source file has been selected.")

        step = _current_step(state)

        source = await asyncio.to_thread(
            read_configured_workspace_file,
            workspace_root=state["workspace_root"],
            relative_path=selected_file,
        )

        additional_context = json.dumps(
            {
                "task_description": state["task_description"],
                "step_number": step["step_number"],
                "step_title": step["title"],
                "acceptance_criteria": step["acceptance_criteria"],
                "verification_actions": step["verification_actions"],
            },
            ensure_ascii=False,
        )[:12_000]

        request = CodePatchRequest(
            workspace_root=state["workspace_root"],
            relative_path=source.relative_path,
            expected_sha256=source.sha256,
            finding_title=step["title"],
            finding_evidence=(
                source.content[:4_000]
                or "The selected source file is empty."
            ),
            finding_explanation=step["description"],
            suggested_fix=(
                "Implement this planned change while satisfying "
                "the following acceptance criteria: "
                + "; ".join(step["acceptance_criteria"])
            )[:8_000],
            additional_context=additional_context,
        )

        proposal = await generate_code_patch(request)

        return {
            "patch_proposal": proposal.model_dump(mode="json"),
            "status": "awaiting_patch_approval",
            "pending_action": {
                "kind": "patch_approval",
                "step_number": step["step_number"],
                "message": (
                    "Review the proposed code diff. "
                    "Approve it to modify the selected file."
                ),
            },
            "message": (
                f"Patch proposal generated for {source.relative_path}. "
                "Approval is required before applying it."
            ),
            "error_message": None,
        }

    except (
        AIProviderError,
        CodePatchError,
        WorkspaceReadError,
    ) as exc:
        logger.warning("Agent patch generation failed: %s", exc)

        return _failure(
            "Could not generate a patch for the selected file.",
            str(exc),
        )

    except Exception as exc:
        logger.exception("Unexpected agent patch-generation failure.")

        return _failure(
            "Unexpected error while generating the patch.",
            str(exc),
        )



async def analyze_test_failure_node(
    state: AgentWorkflowState,
) -> dict:
    """Diagnose a failed test run and identify candidate source files."""

    attempts = state.get("repair_attempts", 0)
    max_attempts = state.get("max_repair_attempts", 2)

    if attempts >= max_attempts:
        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "The maximum correction attempts have been exhausted.",
        }

    test_result = state.get("test_result")

    if not isinstance(test_result, dict):
        return _failure(
            "Test failure analysis cannot start without a test result."
        )

    try:
        analysis = await analyze_test_failure(
            workspace_root=state["workspace_root"],
            task_description=state["task_description"],
            test_result=test_result,
        )

        analysis_data = analysis.model_dump(mode="json")

        if not analysis.candidate_files:
            return {
                "failure_analysis": analysis_data,
                "repair_mode": False,
                "status": "tests_failed",
                "pending_action": None,
                "message": (
                    "The test failure was analyzed, but no verified "
                    "candidate source files were identified. "
                    "Manual investigation is required."
                ),
                "error_message": analysis.root_cause,
            }

        return {
            "failure_analysis": analysis_data,
            "repair_mode": True,
            "status": "awaiting_repair_file_selection",
            "pending_action": {
                "kind": "repair_file_selection",
                "message": (
                    "Review the failure diagnosis and select the "
                    "source file to correct."
                ),
                "available_files": analysis.candidate_files,
            },
            "message": (
                f"Test failure analyzed. "
                f"Correction attempt {attempts + 1} "
                f"of {max_attempts} is available."
            ),
            "error_message": None,
        }

    except (AIProviderError, TestFailureAnalysisError) as exc:
        logger.warning("Test failure analysis failed: %s", exc)

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Automatic test failure analysis could not complete.",
            "error_message": str(exc),
        }

    except WorkspaceReadError as exc:
        logger.warning("Could not inspect workspace for failure analysis: %s", exc)

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "The project could not be inspected for failure analysis.",
            "error_message": str(exc),
        }

    except Exception:
        logger.exception("Unexpected test failure analysis error.")

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Unexpected error during test failure analysis.",
            "error_message": "The failure analysis could not be completed.",
        }


def select_repair_file_node(
    state: AgentWorkflowState,
) -> dict:
    """Require the user to select a verified file for correction."""

    pending = state.get("pending_action") or {}
    available_files = pending.get("available_files", [])

    analysis_data = state.get("failure_analysis")

    if not isinstance(analysis_data, dict):
        return _failure("The failure analysis is unavailable.")

    analysis = TestFailureAnalysis.model_validate(analysis_data)

    decision = interrupt(
        {
            "kind": "repair_file_selection",
            "message": pending.get(
                "message",
                "Select a file to correct.",
            ),
            "analysis": analysis.model_dump(mode="json"),
            "available_files": available_files,
        },
        response_schema=AgentWorkflowResumeRequest,
    )

    if decision.action == "cancel":
        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Correction was cancelled by the user.",
        }

    if decision.action != "select_file":
        return _failure("A valid correction file selection is required.")

    selected_file = decision.selected_file

    if (
        selected_file not in available_files
        or selected_file not in analysis.candidate_files
    ):
        return _failure(
            "The selected file is not an approved candidate."
        )

    return {
        "selected_file": selected_file,
        "repair_mode": True,
        "status": "running",
        "pending_action": None,
        "message": (
            f"Selected {selected_file}. "
            "Generating a corrective patch."
        ),
    }


async def generate_repair_patch_node(
    state: AgentWorkflowState,
) -> dict:
    """Generate a correction proposal based on test failure analysis."""

    try:
        selected_file = state.get("selected_file")

        if not selected_file:
            raise ValueError("No correction file has been selected.")

        analysis_data = state.get("failure_analysis")

        if not isinstance(analysis_data, dict):
            raise ValueError("Test failure analysis is unavailable.")

        analysis = TestFailureAnalysis.model_validate(analysis_data)

        if selected_file not in analysis.candidate_files:
            raise ValueError("The selected file is not a verified candidate.")

        source = await asyncio.to_thread(
            read_configured_workspace_file,
            workspace_root=state["workspace_root"],
            relative_path=selected_file,
        )

        test_result = state.get("test_result") or {}

        stdout = sanitize_test_output(test_result.get("stdout"))
        stderr = sanitize_test_output(test_result.get("stderr"))

        failure_output = "\n".join(
            part
            for part in [
                sanitize_test_output(test_result.get("stderr")),
                sanitize_test_output(test_result.get("stdout")),
            ]
            if part
        )

        failure_evidence = (
            failure_output
            or analysis.explanation
            or source.content[:4000]
        )[:4000]

        additional_context = json.dumps(
            {
                "task_description": state["task_description"],
                "repair_attempt": state.get("repair_attempts", 0) + 1,
                "failure_analysis": analysis.model_dump(mode="json"),
                "test_runner": test_result.get("runner"),
                "test_status": test_result.get("status"),
                "exit_code": test_result.get("exit_code"),
                "failed_test_output": failure_evidence,
                "source_excerpt": source.content[:3500],
                "instruction": (
                    "Fix the diagnosed test failure with the smallest "
                    "reasonable change. Preserve unrelated behavior."
                ),
            },
            ensure_ascii=False,
        )[:12_000]

        request = CodePatchRequest(
            workspace_root=state["workspace_root"],
            relative_path=source.relative_path,
            expected_sha256=source.sha256,
            finding_title=analysis.summary[:200],
            finding_evidence=failure_evidence,
            finding_explanation=(
                f"Likely root cause: {analysis.root_cause}\n\n"
                f"Technical explanation: {analysis.explanation}"
            )[:8000],
            suggested_fix=analysis.suggested_fix[:8000],
            additional_context=additional_context,
        )

        proposal = await generate_code_patch(request)

        return {
            "patch_proposal": proposal.model_dump(mode="json"),
            "status": "awaiting_patch_approval",
            "pending_action": {
                "kind": "patch_approval",
                "message": (
                    "Review the proposed correction. "
                    "Approving it will modify the selected file."
                ),
            },
            "message": (
                f"Corrective patch generated for {source.relative_path}. "
                "Your approval is required."
            ),
            "error_message": None,
        }

    except (
        AIProviderError,
        CodePatchError,
        WorkspaceReadError,
    ) as exc:
        logger.warning("Corrective patch generation failed: %s", exc)

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Could not generate a corrective patch.",
            "error_message": str(exc),
        }

    except Exception:
        logger.exception("Unexpected corrective patch generation error.")

        return {
            "status": "failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Unexpected error during corrective patch generation.",
            "error_message": "The corrective patch could not be generated.",
        }



async def approve_and_apply_patch_node(
    state: AgentWorkflowState,
) -> dict:
    """Request approval and apply a patch only after explicit consent."""

    proposal = CodePatchResponse.model_validate(
        state["patch_proposal"]
    )

    pending = state.get("pending_action") or {}
    repair_mode = bool(state.get("repair_mode", False))

    decision = interrupt(
        {
            "kind": "patch_approval",
            "message": pending.get(
                "message",
                "Do you approve this code patch?",
            ),
            "step_number": pending.get("step_number"),
            "relative_path": proposal.relative_path,
            "summary": proposal.summary,
            "diff": proposal.diff,
            "proposed_sha256": proposal.proposed_sha256,
            "repair_mode": repair_mode,
            "repair_attempt": (
                state.get("repair_attempts", 0) + 1
                if repair_mode
                else None
            ),
        },
        response_schema=AgentWorkflowResumeRequest,
    )

    if decision.action == "cancel":
        return {
            "status": "cancelled",
            "repair_mode": False,
            "pending_action": None,
            "message": "Workflow cancelled before applying the patch.",
        }

    if decision.action == "reject":
        if repair_mode:
            return {
                "status": "tests_failed",
                "repair_mode": False,
                "selected_file": None,
                "pending_action": None,
                "message": (
                    "The repair patch was rejected. "
                    "The original test failure remains unresolved."
                ),
            }

        index = state.get("current_step_index", 0)
        step = _current_step(state)
        step_number = step["step_number"]

        skipped = list(state.get("skipped_steps", []))
        _add_unique(skipped, step_number)

        return {
            "skipped_steps": skipped,
            "current_step_index": index + 1,
            "selected_file": None,
            "pending_action": None,
            "status": "running",
            "message": (
                f"Patch for step {step_number} was rejected. "
                "The workflow will check the next step."
            ),
        }

    if decision.action != "approve":
        return _failure("Invalid decision for patch approval.")

    step_number = None
    index = state.get("current_step_index", 0)

    if not repair_mode:
        step = _current_step(state)
        step_number = step["step_number"]

    try:
        request = CodePatchApplyRequest(
            workspace_root=state["workspace_root"],
            relative_path=proposal.relative_path,
            expected_sha256=proposal.original_sha256,
            proposed_sha256=proposal.proposed_sha256,
            proposed_code=proposal.proposed_code,
            approval="apply",
        )

        result = await apply_code_patch(request)

        applied_patches = list(
            state.get("applied_patches", [])
        )

        applied_patches.append({
            "relative_path": result.relative_path,
            "backup_id": result.backup_id,
            "original_sha256": result.original_sha256,
            "applied_sha256": result.applied_sha256,
        })

        if repair_mode:
            attempts = state.get("repair_attempts", 0) + 1

            return {
                "applied_patches": applied_patches,
                "repair_attempts": attempts,
                "repair_mode": True,
                "selected_file": None,
                "pending_action": {
                    "kind": "test_approval",
                    "message": (
                        f"Repair attempt {attempts} was applied to "
                        f"{result.relative_path}. Approve rerunning tests."
                    ),
                },
                "status": "awaiting_test_approval",
                "message": (
                    "The approved repair patch was applied. "
                    "Tests must be rerun to verify the correction."
                ),
                "error_message": None,
            }

        completed_steps = list(
            state.get("completed_steps", [])
        )
        _add_unique(completed_steps, step_number)

        return {
            "applied_patches": applied_patches,
            "completed_steps": completed_steps,
            "current_step_index": index + 1,
            "selected_file": None,
            "pending_action": None,
            "status": "running",
            "message": (
                f"Step {step_number} completed. "
                f"Patch applied to {result.relative_path}."
            ),
        }

    except Exception as exc:
        logger.exception("Approved patch application failed.")

        return _failure(
            "The approved patch could not be applied.",
            str(exc),
        )



async def approve_and_run_tests_node(
    state: AgentWorkflowState,
) -> dict:
    """Request test approval and execute the configured test runner."""

    repair_mode = bool(state.get("repair_mode", False))
    repair_attempts = state.get("repair_attempts", 0)

    decision = interrupt(
        {
            "kind": "test_approval",
            "message": (
                "A repair patch has been applied. "
                "Do you approve rerunning the tests?"
                if repair_mode
                else (
                    "All implementation steps have been processed. "
                    "Do you approve running the project's test suite?"
                )
            ),
            "runner": state["test_runner"],
            "timeout_seconds": state["test_timeout_seconds"],
            "completed_steps": state.get("completed_steps", []),
            "manual_steps": state.get("manual_steps", []),
            "skipped_steps": state.get("skipped_steps", []),
            "repair_mode": repair_mode,
            "repair_attempts": repair_attempts,
        },
        response_schema=AgentWorkflowResumeRequest,
    )

    if decision.action == "cancel":
        return {
            "status": "cancelled",
            "repair_mode": False,
            "pending_action": None,
            "message": "Workflow cancelled before test execution.",
        }

    if decision.action == "reject":
        return {
            "status": "completed_without_tests",
            "repair_mode": False,
            "pending_action": None,
            "message": (
                "The repair patch was applied, but test rerunning "
                "was declined. The correction remains unverified."
                if repair_mode
                else (
                    "Implementation workflow finished. "
                    "Test execution was declined."
                )
            ),
        }

    if decision.action != "approve":
        return _failure(
            "Invalid decision for test execution approval."
        )

    try:
        result = await asyncio.to_thread(
            execute_tests,
            workspace_root=state["workspace_root"],
            runner=state["test_runner"],
            timeout_seconds=state["test_timeout_seconds"],
        )

        await record_agent_test_run(result)

        response = TestExecutionResponse(
            **asdict(result)
        )

        if response.status == "passed":
            return {
                "test_result": response.model_dump(mode="json"),
                "repair_mode": False,
                "status": "completed",
                "pending_action": None,
                "message": (
                    "The repair was verified successfully."
                    if repair_mode
                    else (
                        "Implementation workflow and tests "
                        "completed successfully."
                    )
                ),
            }

        # Keep the graph active so it can analyze this failure.
        return {
            "test_result": response.model_dump(mode="json"),
            "status": "running",
            "pending_action": None,
            "message": (
                "Tests still fail after the repair."
                if repair_mode
                else (
                    "The test suite failed. "
                    "Analyzing the failure."
                )
            ),
        }

    except TestExecutionError as exc:
        logger.warning("Agent test execution failed: %s", exc)

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "The test runner could not complete successfully.",
            "error_message": str(exc),
        }

    except Exception as exc:
        logger.exception("Unexpected agent test execution failure.")

        return {
            "status": "tests_failed",
            "repair_mode": False,
            "pending_action": None,
            "message": "Unexpected error during test execution.",
            "error_message": str(exc),
        }
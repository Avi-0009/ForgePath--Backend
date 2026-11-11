
# ForgePath Execution Surface Audit

## 1. Audit Scope

This audit covers all identified project-code execution paths, workspace boundaries, patch application, and process management.

The audit is based on the current main branch.

## 2. Project Execution Entry Points

### 2.1 Standalone Test Execution

File:
app/api/v1/routes/execution.py

Endpoint:
POST /api/v1/test-execution/run

Execution:

- Requires authentication.
- Requires explicit approval.
- Accepts only pytest, npm, or go.
- Delegates execution to execute_tests().

### 2.2 Agent-Driven Test Execution

File:
app/services/agent_orchestration/nodes.py

Execution:

- Runs through the LangGraph workflow.
- Requires explicit test approval.
- Uses the same execute_tests() function.
- Repair attempts require separate test approval.
- Does not accept arbitrary shell commands from the AI.

### 2.3 Central Execution Engine

File:
app/services/test_execution/runner.py

Current protections:

- Uses predefined runner commands.
- Uses shell=False.
- Filters inherited environment variables.
- Enforces execution time limits.
- Captures bounded stdout and stderr.
- Attempts process-tree termination.
- Uses an allowlisted workspace.

Important limitation:

The project code runs under the backend process's operating-system permissions. This is not an OS-level sandbox.

The Windows npm runner invokes cmd.exe /c npm test, allowing the project's configured npm test script to execute.

## 3. Workspace Security

File:
app/services/workspace.py

Existing protections:

- Configured workspace allowlist.
- Parent-directory traversal rejection.
- Symbolic-link and junction checks.
- Relative path validation.
- Sensitive-file exclusions for workspace reading.

File:
app/services/workspace_ingestion.py

Existing protections:

- Skips symbolic links and junctions.
- Enforces file-count and file-size limits.
- Enforces directory-depth and total-size limits.
- Excludes configured directories and sensitive files.

Important limitation:

These application-level reading protections do not prevent an executed test process from directly accessing files that its operating-system identity can read.

## 4. Patch Application Security

File:
app/services/code_patch_apply.py

Existing protections:

- Explicit patch approval.
- Source hash verification.
- Original-file backups.
- Atomic file replacement.
- Post-write hash verification.
- Backup integrity validation during rollback.

Patch application is a file-modification path, not an independent project-command execution engine.

## 5. Process and Resource Management

Existing protections:

- Execution timeout between 1 and 600 seconds.
- Maximum output capture of 64 KB per stream.
- Separate stdout and stderr readers.
- Windows taskkill process-tree termination on timeout.
- POSIX process-group termination.

Missing protections:

- OS-enforced memory limits.
- CPU quotas.
- Process-count limits.
- Disk usage limits.
- Default network restrictions.
- Reliable Windows Job Object lifecycle management.

## 6. Main Security Gaps

1. No operating-system-enforced execution isolation.
2. Executed project code can access host files available to the backend identity.
3. Backend credentials are filtered from the child environment, but host files containing secrets remain accessible.
4. Network access is not restricted.
5. Resource limits are incomplete.
6. Process-tree cleanup needs stronger guarantees.
7. Filesystem race-condition protection requires further review.
8. The npm runner invokes a Windows command interpreter.

## 7. Milestone 9 Implementation Mapping

Chunk 1: Security audit and execution policy.
Chunk 2: Command and workspace validation.
Chunk 3: Isolated command execution.
Chunk 4: Process and resource protection.
Chunk 5: Security testing and final verification.

## 8. Security Principle

Project code is untrusted.

Explicit user approval authorizes a requested operation, but does not grant that code unrestricted access to the host machine.

The execution system must fail closed when required isolation or security controls are unavailable.


# ForgePath Command Execution Security Policy

## 1. Purpose

Define the security requirements for executing project code through ForgePath.

ForgePath is a local-first AI coding agent. Project code must be treated as untrusted, even when execution is explicitly approved by the user.

## 2. Approved Execution Runners

The supported runners are:

- pytest
- npm
- go

Only predefined commands and arguments are permitted.

Arbitrary shell commands, dynamically generated executable paths, and unrestricted command arguments must not be accepted from API requests or AI-generated plans.

## 3. Execution Approval

- Standalone test execution requires explicit user approval.
- Agent-driven test execution requires explicit approval at the workflow interruption.
- Repair test reruns require separate approval.
- AI-generated patches must never execute automatically.

## 4. Workspace Boundaries

- Execution must be scoped to an explicitly authorized workspace.
- Parent-directory traversal must be rejected.
- Symbolic links and Windows junctions must be checked.
- Filesystem validation must account for path changes and race conditions.
- Access outside the authorized project must be denied.

## 5. Sensitive Information

Executed project code must not receive access to:

- Backend environment variables and API credentials.
- Database credentials and JWT secrets.
- SSH private keys.
- Cloud-provider credentials.
- User profile secrets.
- Other unrelated host files.

Sensitive files and directories must be excluded from project inspection and execution access.

Examples include `.env`, `.env.*`, `.ssh`, `.aws`, `.azure`, `.npmrc`, `.pypirc`, `.netrc`, private keys, and credential files.

## 6. Execution Isolation

Project code must execute in an operating-system-enforced isolated environment.

The execution environment must:

- Run without administrative privileges.
- Restrict host filesystem access.
- Prevent access to backend application secrets.
- Restrict network access by default.
- Avoid unrestricted access to host processes.
- Fail closed if the isolation mechanism is unavailable.

There must be no automatic fallback to unrestricted host execution.

## 7. Resource Protection

Every execution must have bounded:

- Execution duration.
- CPU consumption.
- Memory consumption.
- Process count.
- Disk usage.
- Standard output and standard error.

Timeouts and failures must trigger process-tree cleanup.

## 8. Patch Safety

- Patch application requires explicit approval.
- Original source files must be backed up.
- Source hashes must be verified before modification.
- Rollback must verify backup integrity.
- Test execution must not silently apply additional changes to the host workspace.

## 9. Audit and Telemetry

Only safe execution metadata may be persisted.

Prompts, source code, secrets, command output, and environment values must not be stored in usage telemetry.

## 10. Security Verification

Security tests must cover:

- Command injection attempts.
- Unauthorized workspace access.
- Path traversal.
- Symlink and junction attacks.
- Sensitive-file access.
- Environment-variable leakage.
- Process-tree termination.
- CPU, memory, and disk limits.
- Network restrictions.
- Sandbox failure and fallback behavior.

All security requirements must be verified before Milestone 9 is considered complete.

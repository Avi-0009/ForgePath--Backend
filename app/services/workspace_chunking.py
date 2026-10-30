import hashlib
import re

from dataclasses import dataclass

from app.services.workspace_ingestion import (
    IngestedWorkspaceDocument,
)


DEFAULT_MAX_CHUNK_CHARS = 4_000
DEFAULT_CHUNK_OVERLAP_CHARS = 400

_DECLARATION_PATTERN = re.compile(
    r"^\s*(?:(?:export|default|async|pub|public|private|protected|"
    r"internal|static|abstract|final)\s+)*"
    r"(?:def|class|function|interface|type|enum|struct|trait|"
    r"impl|fn|func|namespace|module)\b"
)

_VARIABLE_DECLARATION_PATTERN = re.compile(
    r"^\s*(?:(?:export)\s+)?(?:const|let|var)\s+\w+\s*="
)


@dataclass(frozen=True, slots=True)
class WorkspaceChunk:
    """A bounded source-code or documentation chunk with provenance."""

    workspace_root: str
    relative_path: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    document_sha256: str
    chunk_sha256: str


def _is_preferred_boundary(
    current_line: str,
    previous_line: str,
) -> bool:
    """Identify boundaries that are useful for code and documentation."""

    if not current_line.strip():
        return True

    if not previous_line.strip():
        return True

    return bool(
        _DECLARATION_PATTERN.match(current_line)
        or _VARIABLE_DECLARATION_PATTERN.match(current_line)
    )


def _find_chunk_end(
    lines: list[str],
    start: int,
    max_chars: int,
) -> int:
    """Choose a line boundary while keeping the chunk within its limit."""

    cursor = start
    char_count = 0
    candidates: list[tuple[int, int]] = []

    while cursor < len(lines):
        line_length = len(lines[cursor])

        if char_count + line_length > max_chars:
            break

        if cursor > start and _is_preferred_boundary(
            lines[cursor],
            lines[cursor - 1],
        ):
            candidates.append((char_count, cursor))

        char_count += line_length
        cursor += 1

    if cursor < len(lines):
        minimum_boundary_size = max_chars * 0.60

        eligible = [
            candidate
            for candidate in candidates
            if candidate[0] >= minimum_boundary_size
        ]

        if eligible:
            return eligible[-1][1]

    return cursor


def _build_chunk(
    document: IngestedWorkspaceDocument,
    *,
    chunk_index: int,
    start_line: int,
    end_line: int,
    content: str,
) -> WorkspaceChunk:
    """Build a chunk and calculate a deterministic content hash."""

    chunk_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return WorkspaceChunk(
        workspace_root=document.workspace_root,
        relative_path=document.relative_path,
        language=document.language,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        content=content,
        document_sha256=document.sha256,
        chunk_sha256=chunk_hash,
    )


def chunk_document(
    document: IngestedWorkspaceDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> tuple[WorkspaceChunk, ...]:
    """
    Split an ingested document into bounded, line-aware chunks.

    Preferred boundaries include blank lines and common declarations.
    Overlap uses complete lines where possible to preserve code structure.

    Very long individual lines are split into character windows because
    preserving the maximum chunk size takes priority in that case.

    This function does not call embedding providers or persist content.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero.")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be non-negative and smaller than max_chars."
        )

    if not document.content:
        return ()

    lines = document.content.splitlines(keepends=True)

    if not lines:
        return ()

    chunks: list[WorkspaceChunk] = []
    start = 0

    while start < len(lines):
        # A single line may be larger than the entire chunk budget.
        if len(lines[start]) > max_chars:
            line = lines[start]
            step = max_chars - overlap_chars
            position = 0

            while position < len(line):
                segment = line[position : position + max_chars]

                chunks.append(
                    _build_chunk(
                        document,
                        chunk_index=len(chunks),
                        start_line=start + 1,
                        end_line=start + 1,
                        content=segment,
                    )
                )

                if position + max_chars >= len(line):
                    break

                position += step

            start += 1
            continue

        end = _find_chunk_end(
            lines,
            start,
            max_chars,
        )

        if end <= start:
            raise RuntimeError(
                "Chunking failed to advance through the document."
            )

        content = "".join(lines[start:end])

        chunks.append(
            _build_chunk(
                document,
                chunk_index=len(chunks),
                start_line=start + 1,
                end_line=end,
                content=content,
            )
        )

        if end >= len(lines):
            break

        # Reuse complete lines from the previous chunk where possible.
        overlap_start = end
        overlap_size = 0

        while overlap_start > start:
            previous_line_length = len(lines[overlap_start - 1])

            if overlap_size + previous_line_length > overlap_chars:
                break

            overlap_size += previous_line_length
            overlap_start -= 1

        # Always advance, even when no complete line fits the overlap.
        start = (
            overlap_start
            if overlap_start < end
            else end
        )

    return tuple(chunks)
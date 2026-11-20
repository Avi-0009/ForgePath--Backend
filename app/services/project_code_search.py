"""Deterministic, bounded text search over an existing private snapshot.

No provider calls, indexing, filesystem paths from clients, or execution.
"""

from __future__ import annotations

import re


MAX_QUERY_CHARS = 200
MAX_RESULTS = 20
MAX_EXCERPT_CHARS = 1200
STOP_WORDS = {"a", "an", "the", "is", "are", "of", "to", "in", "for", "with",
              "where", "how", "what", "which", "does", "do", "show", "me"}


def search_snapshot_sources(
    sources: list[tuple[str, str]],
    query: str,
    limit: int = 6,
) -> list[dict]:
    """Return ranked, source-quoted excerpts with stable line numbers."""
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError("Search limit must be between 1 and 20")
    if not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise ValueError("Search query must contain 1-200 characters")

    terms = [term for term in dict.fromkeys(re.findall(r"[\w-]{2,}", query.casefold()))
             if term not in STOP_WORDS][:12]
    if not terms:
        terms = [query.strip().casefold()]

    matches: list[tuple[int, str, int, dict]] = []
    for path, content in sources:
        lines = content.splitlines()
        path_lower = path.casefold()
        path_score = sum(2 for term in terms if term in path_lower)
        file_matches = 0
        for index, line in enumerate(lines):
            lowered = line.casefold()
            matched = sum(1 for term in terms if term in lowered)
            if not matched and not path_score:
                continue
            score = matched * 3 + path_score
            start = max(0, index - 2)
            end = min(len(lines), index + 3)
            excerpt = "\n".join(lines[start:end])[:MAX_EXCERPT_CHARS]
            matches.append((
                -score, path, index,
                {
                    "relative_path": path,
                    "language": path.rsplit(".", 1)[-1].lower() if "." in path else "",
                    "chunk_index": index,
                    "start_line": start + 1,
                    "end_line": min(end, start + len(excerpt.splitlines())),
                    "content": excerpt,
                    "distance": None,
                },
            ))
            file_matches += 1
            # Bound per-file memory even for a very repetitive source file.
            if file_matches >= 10:
                break

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in matches[:limit]]
"""Bounded, provenance-rich context for question-first project investigations.

Everything here operates on immutable, owner-authorized snapshot text. It does
not read local paths, call an AI provider, or write to the user's repository.
Semantic hits must already be verified against that snapshot by the caller.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from typing import Iterable

from app.services.project_investigation_context import (
    MAX_INVESTIGATION_FILES,
    _query_terms,
    select_investigation_files,
)
from app.services.project_reference_integrity import resolved_source_graph


MAX_CONTEXT_CHARS = 26_000
MAX_EXCERPT_CHARS = 3_200
MAX_SEMANTIC_HITS = 8


def _choose_excerpt(content: str, terms: set[str], limit: int, hint: str = "") -> tuple[str, int, int]:
    """Extract a verbatim, line-numbered excerpt; never synthesize source text."""
    lines = content.splitlines(keepends=True)
    if not lines or limit <= 0:
        return "", 1, 1
    if hint and hint in content:
        center = content[:content.index(hint)].count("\n")
    else:
        scores = [sum(word in line.casefold() for word in terms) for line in lines]
        center = max(range(len(lines)), key=lambda pos: sum(scores[pos:pos + 12]))
    start = max(0, center - 3)
    excerpt = "".join(lines[start:])[:limit]
    end = start + excerpt.count("\n") + (0 if excerpt.endswith("\n") else 1)
    return excerpt, start + 1, max(start + 1, end)


def build_investigation_context(
    sources: Iterable[tuple[str, str]],
    question: str,
    max_files: int = 5,
    *,
    semantic_hits: Iterable[dict] = (),
    continuity_paths: Iterable[str] = (),
) -> dict:
    """Combine lexical relevance, verified import edges, and optional vectors.

    `semantic_hits` is an optional sequence of {relative_path, content}; each
    hit is rechecked against the source text before influencing selection.
    No index search or provider call takes place in this function.
    """
    if not 1 <= max_files <= MAX_INVESTIGATION_FILES:
        raise ValueError("File count must be between 1 and 10")
    if not question.strip() or len(question) > 2000:
        raise ValueError("Question must contain 1–2000 characters")

    source_list = list(sources)
    source_map = dict(source_list)
    graph = resolved_source_graph(source_list)
    callers: dict[str, set[str]] = {path: set() for path in source_map}
    for source_path, imported_paths in graph.items():
        for imported in imported_paths:
            callers[imported].add(source_path)

    lexical = select_investigation_files(source_list, question, min(10, MAX_INVESTIGATION_FILES))
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    def add(path: str, score: float, reason: str) -> None:
        if path in source_map and source_map[path].strip() and score > scores.get(path, -1):
            scores[path] = score
            reasons[path] = reason

    for rank, item in enumerate(lexical):
        add(str(item["path"]), max(15, 120 - 9 * rank), "lexical relevance")

    hints: dict[str, str] = {}
    verified_semantic = 0
    for rank, hit in enumerate(list(semantic_hits)[:MAX_SEMANTIC_HITS]):
        path, excerpt = hit.get("relative_path"), hit.get("content")
        if not isinstance(path, str) or not isinstance(excerpt, str):
            continue
        if path not in source_map or not excerpt.strip() or excerpt not in source_map[path]:
            continue
        verified_semantic += 1
        hints.setdefault(path, excerpt[:300])
        add(path, max(50, 135 - rank * 9), "verified semantic match")

    # Continuity paths are merely pointers to the current saved snapshot.
    # Old conversation evidence is NEVER accepted as evidence without rechecking.
    for rank, path in enumerate(list(continuity_paths)[:6]):
        if isinstance(path, str):
            add(path, max(35, 88 - 6 * rank), "source previously selected in this conversation")

    # Expand only one verified import hop. Dependencies are usually the missing
    # implementation; reverse edges can reveal callers and integration tests.
    anchors = sorted(scores, key=lambda path: (-scores[path], path))[:6]
    for rank, anchor in enumerate(anchors):
        for dependency in sorted(graph.get(anchor, ())):
            add(dependency, max(25, 112 - rank * 9), f"imported by {anchor}")
        for caller in sorted(callers.get(anchor, ())):
            add(caller, max(20, 95 - rank * 8), f"imports {anchor}")

    remaining = set(scores)
    selected: list[str] = []
    stems: set[str] = set()
    parents: Counter[str] = Counter()
    while remaining and len(selected) < max_files:
        def rank(path: str) -> tuple[float, float, str]:
            item = PurePosixPath(path)
            penalty = (14 if item.stem.casefold() in stems else 0)
            penalty += min(18, parents[str(item.parent)] * 7)
            return scores[path] - penalty, scores[path], path

        winner = max(sorted(remaining), key=rank)
        remaining.remove(winner)
        selected.append(winner)
        item = PurePosixPath(winner)
        stems.add(item.stem.casefold())
        parents[str(item.parent)] += 1

    terms = _query_terms(question)
    budget = MAX_CONTEXT_CHARS
    files = []
    for path in selected:
        if budget <= 0:
            break
        excerpt, start, end = _choose_excerpt(
            source_map[path], terms, min(MAX_EXCERPT_CHARS, budget), hints.get(path, ""),
        )
        if not excerpt:
            continue
        budget -= len(excerpt)
        files.append({
            "path": path,
            "reason": reasons[path],
            "score": round(scores[path], 2),
            "start_line": start,
            "end_line": end,
            "excerpt": excerpt,
            "excerpt_chars": len(excerpt),
        })

    return {
        "files": files,
        "total_files": len(source_list),
        "graph_edges": sum(map(len, graph.values())),
        "semantic_hits_verified": verified_semantic,
        "candidate_files": len(scores),
        "context_chars": MAX_CONTEXT_CHARS - budget,
        "estimated_tokens": (MAX_CONTEXT_CHARS - budget + 3) // 4,
        "selection_method": "lexical+imports+callers" + ("+semantic" if verified_semantic else ""),
    }


def public_context_preview(context: dict) -> dict:
    """Return relevant paths and provenance without transmitting source text."""
    return {
        "selected_files": [{key: value for key, value in item.items() if key != "excerpt"}
                           for item in context["files"]],
        "total_files": context["total_files"],
        "graph_edges": context["graph_edges"],
        "semantic_hits_verified": context["semantic_hits_verified"],
        "candidate_files": context["candidate_files"],
        "context_chars": context["context_chars"],
        "estimated_tokens": context["estimated_tokens"],
        "selection_method": context["selection_method"],
    }
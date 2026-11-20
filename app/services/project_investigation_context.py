"""Owner-scoped snapshot retrieval using lexical, symbol and import signals.

This is a deterministic, zero-provider-call hybrid fallback; neural embeddings
require a separate, opt-in, project-owned index.
"""

from __future__ import annotations

import math
import re
import io
import tokenize
from collections import Counter
from pathlib import PurePosixPath

MAX_INVESTIGATION_FILES = 10
MAX_SOURCE_CHARS = 30_000
MAX_FILE_EXCERPT = 3_600

_STOP = {
    "about", "all", "and", "any", "api", "app", "application", "are", "can", "check", "code", "could",
    "does", "file", "files", "find", "flow", "for", "from", "how", "into",
    "implementation", "investigate", "its", "our", "please", "project", "related", "responsible", "show",
    "source", "the", "their", "these", "this", "what", "when", "where",
    "which", "why", "with", "your",
}
_EXPANSIONS = {
    "authentication": "auth authenticate login jwt current_user",
    "authorization": "authorize permission dependency current_user",
    "access": "auth bearer jwt",
    "tokens": "token jwt bearer refresh",
    "token": "jwt bearer refresh",
    "refreshing": "refresh rotate session",
    "expired": "expire expiry expiration",
    "sessions": "session refresh revoke",
    "validating": "validate verify decode",
    "endpoints": "endpoint route router dependency",
    "protecting": "protected permission security current_user",
    "tests": "test pytest",
}
_DECLARATION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|interface|func|fn)\s+([\w]+)", re.M
)
_PY_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M
)


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    words = [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) >= 3]
    return [word[:-3] + "y" if word.endswith("ies") and len(word) > 5 else word
            for word in words]


def _code_tokens(path: str, content: str) -> list[str]:
    if not path.endswith((".py", ".pyi")):
        return _tokens(content)
    try:
        names = (item.string for item in tokenize.generate_tokens(io.StringIO(content).readline)
                 if item.type == tokenize.NAME)
        return _tokens(" ".join(names))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _tokens(content)


def _query_terms(question: str) -> set[str]:
    terms = {term for term in _tokens(question) if term not in _STOP}
    for term in tuple(terms):
        terms.update(_tokens(_EXPANSIONS.get(term, "")))
        if term.endswith("ing") and len(term) > 6:
            terms.add(term[:-3])
        if term.endswith("s") and len(term) > 4:
            terms.add(term[:-1])
    return terms


def _excerpt(content: str, terms: set[str], limit: int) -> tuple[str, int]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return "", 1
    counts = [sum(term in line.casefold() for term in terms) for line in lines]
    best = max(range(len(lines)), key=lambda index: sum(counts[index:index + 12]))
    start = max(0, best - 2)
    return "".join(lines[start:])[:limit], start + 1


def select_investigation_files(
    sources: list[tuple[str, str]], question: str, limit: int = 5,
) -> list[dict[str, str | int | float]]:
    """Bounded relevance ranking with one-hop Python import expansion."""
    if not question.strip() or len(question) > 2000:
        raise ValueError("Question must contain 1-2000 characters")
    if not 1 <= limit <= MAX_INVESTIGATION_FILES:
        raise ValueError("File limit must be between 1 and 10")

    terms = _query_terms(question) or set(_tokens(question))
    docs = []
    document_frequency = Counter()
    for path, content in sources:
        counts = Counter(_code_tokens(path, content))
        path_terms = Counter(_tokens(path))
        symbols = Counter(_tokens(" ".join(_DECLARATION.findall(content))))
        document_frequency.update((set(counts) | set(path_terms) | set(symbols)) & terms)
        docs.append((path, content, counts, path_terms, symbols))

    total = max(len(docs), 1)
    ranked = []
    for path, content, counts, path_terms, symbols in docs:
        score = 0.0
        for term in terms:
            idf = math.log(1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            tf = counts[term]
            score += idf * ((tf * 1.2 / (tf + 1.2) if tf else 0)
                            + 13 * bool(path_terms[term]) + 4 * bool(symbols[term]))
        lower_path = path.casefold()
        if lower_path.startswith(("app/", "src/", "lib/")):
            score *= 1.15
        if lower_path.startswith(("tests/", "test/")):
            score *= 1 if "test" in terms else 0.65
        if lower_path.startswith(("alembic/", "migrations/")):
            score *= 1 if "migration" in terms else 0.2
        ranked.append([score, path, content])

    # A strongly matching endpoint may import its real security implementation.
    modules = {str(PurePosixPath(path).with_suffix("")).replace("/", "."): path
               for path, _ in sources}
    neighbors: dict[str, float] = {}
    for score, path, content in sorted(ranked, key=lambda item: -item[0])[:5]:
        if score <= 0:
            continue
        for match in _PY_IMPORT.finditer(content):
            target = modules.get(match.group(1) or match.group(2))
            if target and target != path:
                neighbors[target] = max(neighbors.get(target, 0), min(score * 0.4, 50))
    for item in ranked:
        item[0] += neighbors.get(item[1], 0)

    # MMR-style diversity: avoid crowding out dependencies with several
    # auth_session.py variants from models, repositories, schemas, and services.
    remaining_candidates = [item for item in ranked if item[0] > 0]
    selected = []
    stems: set[str] = set()
    parent_counts: Counter[str] = Counter()
    while remaining_candidates and len(selected) < limit:
        def adjusted(item: list) -> tuple[float, str]:
            path = PurePosixPath(item[1])
            stem = path.stem.casefold()
            parent = str(path.parent)
            penalty = (55 if stem in stems else 0) + 15 * max(0, parent_counts[parent] - 1)
            return item[0] - penalty, item[1]

        winner = max(remaining_candidates, key=lambda item: (adjusted(item)[0], -len(item[1])))
        remaining_candidates.remove(winner)
        selected.append(winner)
        stems.add(PurePosixPath(winner[1]).stem.casefold())
        parent_counts[str(PurePosixPath(winner[1]).parent)] += 1

    remaining = MAX_SOURCE_CHARS
    result = []
    for score, path, content in selected:
        if remaining <= 0:
            break
        excerpt, start_line = _excerpt(content, terms, min(MAX_FILE_EXCERPT, remaining))
        if not excerpt:
            continue
        remaining -= len(excerpt)
        result.append({
            "path": path, "score": round(score, 3),
            "excerpt": excerpt, "start_line": start_line,
        })
    return result
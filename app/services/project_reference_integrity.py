"""Deterministic, read-only references found in a private project snapshot.

This deliberately does not claim to implement a full compiler or bundler. Only
statically identifiable local references are checked. Bare package names,
configured aliases and computed paths are out of scope and are not reported as
missing. SVG content is never needed: the manifest's path inventory suffices.
"""

from __future__ import annotations

import ast
import posixpath
import re
from pathlib import PurePosixPath
from typing import Iterable


_JS_EXTENSIONS = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
_SOURCE_EXTENSIONS = (".ts", ".tsx", ".d.ts", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css")
_TRACKED_EXTENSIONS = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css",
    ".py", ".svg", ".html", ".md", ".yml", ".yaml", ".toml",
})
_JS_STATIC = re.compile(
    r"(?m)^[ \t]*(?:import|export)[ \t]+(?:type[ \t]+)?"
    r"(?:[^;\n]{0,300}?\bfrom[ \t]+)?[\"'](?P<path>\.{1,2}/[^\"'\n]+)[\"']"
)
_JS_CALL = re.compile(
    r"\b(?:require|import)[ \t]*\([ \t]*[\"'](?P<path>\.{1,2}/[^\"'\n]+)[\"'][ \t]*\)"
)
_CSS_SVG = re.compile(
    r"url\([ \t]*[\"']?(?P<path>(?!data:|https?://)[^\"')\s]+\.svg(?:[?#][^\"')\s]*)?)[\"']?[ \t]*\)",
    re.IGNORECASE,
)
_HTML_SVG = re.compile(
    r"\b(?:src|href)[ \t]*=[ \t]*[\"'](?P<path>(?!data:|https?://)[^\"']+\.svg(?:[?#][^\"']*)?)[\"']",
    re.IGNORECASE,
)
_JS_COMMENTS = re.compile(r"/\*.*?\*/|(?<!:)//[^\n]*", re.DOTALL)


def _without_comments(content: str) -> str:
    return _JS_COMMENTS.sub(
        lambda match: "".join("\n" if char == "\n" else " " for char in match.group()),
        content,
    )


def _candidates(source_path: str, specifier: str) -> list[str]:
    """Return only safe, same-project file candidates in precedence order."""
    requested = specifier.split("?", 1)[0].split("#", 1)[0]
    if not requested or "\\" in requested or "\x00" in requested or ":" in requested:
        return []
    if requested.startswith("/"):
        # Standard Vite public assets are referenced from URL root.
        base = posixpath.normpath("public/" + requested.lstrip("/"))
    else:
        if not (requested.startswith("./") or requested.startswith("../")):
            return []
        base = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), requested))
    if base in {"", ".", ".."} or base.startswith("../") or base.startswith("/"):
        return []
    suffix = PurePosixPath(base).suffix.lower()
    if suffix:
        # A PNG/font/binary may exist locally yet be excluded from the source
        # snapshot by design. Its absence from the index proves nothing.
        if suffix not in _TRACKED_EXTENSIONS:
            return []
        candidates = [base]
        # TypeScript often imports its compiled .js name while storing .ts source.
        if suffix in {".js", ".jsx"}:
            stem = base[: -len(suffix)]
            candidates.extend([stem + ".ts", stem + ".tsx", stem + ".d.ts"])
    else:
        candidates = [base + ext for ext in _SOURCE_EXTENSIONS]
        candidates.extend(base + "/index" + ext for ext in _SOURCE_EXTENSIONS)
    return candidates


def _python_references(path: str, content: str) -> list[tuple[int, str, list[str]]]:
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(content, filename=path)
    except (SyntaxError, ValueError):
        return []

    result: list[tuple[int, str, list[str]]] = []
    folder_parts = list(PurePosixPath(path).parent.parts)
    if folder_parts == ["."]:
        folder_parts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1 or not node.module:
            continue
        # In 'from .utils import x', one dot means the current package.
        upper = node.level - 1
        if upper > len(folder_parts):
            continue
        parent = folder_parts[:len(folder_parts) - upper]
        stem = "/".join([*parent, *node.module.split(".")])
        if not stem or stem.startswith("../"):
            continue
        candidates = [stem + ".py", stem + "/__init__.py"]
        result.append((node.lineno, "." * node.level + node.module, candidates))
    return result


def inspect_snapshot_references(
    sources: Iterable[tuple[str, str]],
    asset_paths: Iterable[str] = (),
    *,
    max_results: int = 200,
) -> dict:
    """Find source-supported local references; no filesystem or network access.

    Returns a bounded list of unresolved references and exact counts for all
    identifiable references. An unresolved reference is a *candidate* issue,
    not proof that a custom loader would fail.
    """
    if not 1 <= max_results <= 500:
        raise ValueError("max_results must be between 1 and 500")
    source_list = list(sources)
    asset_inventory = set(asset_paths)
    available = {path for path, _ in source_list} | asset_inventory
    checked = 0
    resolved = 0
    missing: list[dict] = []

    def record(path: str, line: int, specifier: str, candidates: list[str], kind: str) -> None:
        nonlocal checked, resolved
        if not candidates:
            return
        checked += 1
        target = next((candidate for candidate in candidates if candidate in available), None)
        if target is not None:
            resolved += 1
        elif len(missing) < max_results:
            missing.append({
                "source_path": path,
                "line": line,
                "specifier": specifier,
                "kind": kind,
                "candidate_paths": candidates[:12],
            })

    for path, content in source_list:
        extension = PurePosixPath(path).suffix.lower()
        if extension in _JS_EXTENSIONS:
            searchable = _without_comments(content)
            matches = list(_JS_STATIC.finditer(searchable)) + list(_JS_CALL.finditer(searchable))
            for match in sorted(matches, key=lambda entry: entry.start()):
                specifier = match.group("path")
                record(path, content.count("\n", 0, match.start()) + 1,
                       specifier, _candidates(path, specifier), "import")
        if extension in {".css", ".scss"}:
            for match in _CSS_SVG.finditer(content):
                specifier = match.group("path")
                if not (specifier.startswith("/") or specifier.startswith(".")):
                    specifier = "./" + specifier
                record(path, content.count("\n", 0, match.start()) + 1,
                       match.group("path"), _candidates(path, specifier), "svg_asset")
        if extension in {".tsx", ".jsx", ".html"}:
            for match in _HTML_SVG.finditer(content):
                specifier = match.group("path")
                record(path, content.count("\n", 0, match.start()) + 1,
                       specifier, _candidates(path, specifier), "svg_asset")
        for line, specifier, candidates in _python_references(path, content):
            record(path, line, specifier, candidates, "python_import")

    return {
        "files_in_snapshot": len(source_list),
        "svg_paths_in_manifest": len(asset_inventory),
        "references_checked": checked,
        "references_resolved": resolved,
        "references_unresolved": checked - resolved,
        "results_truncated": checked - resolved > len(missing),
        "missing_candidates": missing,
        "limitations": "Static, relative references only; aliases, external packages, dynamic paths and custom build rules are not validated.",
    }


def resolved_source_graph(sources: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Resolve trustworthy, static source-to-source import edges.

    Unlike the missing-reference report, this function also recognizes Python
    absolute imports *when their target is actually present in this snapshot*.
    Unknown packages and configured aliases never create invented edges.
    Edges never point outside the supplied source list or to SVG metadata.
    """
    source_list = list(sources)
    allowed = {path for path, _ in source_list}
    graph: dict[str, set[str]] = {path: set() for path in allowed}

    def link(source: str, candidates: Iterable[str]) -> None:
        target = next((candidate for candidate in candidates if candidate in allowed), None)
        if target and target != source:
            graph[source].add(target)

    for path, content in source_list:
        extension = PurePosixPath(path).suffix.lower()
        if extension in _JS_EXTENSIONS:
            searchable = _without_comments(content)
            for expression in (_JS_STATIC, _JS_CALL):
                for match in expression.finditer(searchable):
                    link(path, _candidates(path, match.group("path")))
        if extension != ".py":
            continue
        try:
            parsed = ast.parse(content, filename=path)
        except (SyntaxError, ValueError):
            continue
        parent = list(PurePosixPath(path).parent.parts)
        if parent == ["."]:
            parent = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    stem = alias.name.replace(".", "/")
                    link(path, (stem + ".py", stem + "/__init__.py"))
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                if node.level - 1 > len(parent):
                    continue
                root = parent[:len(parent) - node.level + 1]
            else:
                root = []
            module = node.module.split(".") if node.module else []
            base = "/".join([*root, *module])
            if base:
                link(path, (base + ".py", base + "/__init__.py"))
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = "/".join([base, alias.name]) if base else alias.name
                link(path, (candidate + ".py", candidate + "/__init__.py"))
    return graph
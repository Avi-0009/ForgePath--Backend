"""Reference checking requires no filesystem, database, network, or AI calls."""

from app.services.project_reference_integrity import inspect_snapshot_references


def test_js_relative_imports_and_extensionless_resolution():
    result = inspect_snapshot_references([
        ("src/App.tsx", 'import { run } from "./lib/run";\nimport "./styles.css";'),
        ("src/lib/run.ts", "export const run = () => 1"),
        ("src/styles.css", "body { color: red; }"),
    ])
    assert result["references_checked"] == 2
    assert result["references_resolved"] == 2
    assert result["missing_candidates"] == []


def test_svg_inventory_resolves_assets_without_svg_content():
    sources = [
        ("src/App.tsx", 'import logo from "./assets/logo.svg";\nimport missing from "./assets/missing.svg";'),
        ("src/site.css", 'a { background: url("../public/mark.svg"); }'),
    ]
    result = inspect_snapshot_references(sources, ["src/assets/logo.svg", "public/mark.svg"])
    assert result["references_checked"] == 3
    assert result["references_resolved"] == 2
    assert result["references_unresolved"] == 1
    assert result["missing_candidates"][0]["specifier"] == "./assets/missing.svg"
    assert result["missing_candidates"][0]["line"] == 2


def test_html_root_public_asset_and_css_relative_asset():
    result = inspect_snapshot_references([
        ("index.html", '<link href="/favicon.svg" rel="icon">'),
        ("src/style.css", 'body { background-image: url("icons/sprite.svg"); }'),
    ], ["public/favicon.svg", "src/icons/sprite.svg"])
    assert result["references_checked"] == 2
    assert result["references_unresolved"] == 0


def test_does_not_mislabel_packages_aliases_or_comments_as_missing():
    result = inspect_snapshot_references([
        ("src/main.ts", """// import nope from './missing-a'
/* import nope from './missing-b' */
import React from 'react';
import alias from '@/components/Widget';
import valid from './present';
"""),
        ("src/present.ts", "export default 42;"),
    ])
    assert result["references_checked"] == 1
    assert result["references_unresolved"] == 0


def test_typescript_js_extension_can_resolve_ts_source():
    result = inspect_snapshot_references([
        ("src/main.ts", 'import { thing } from "./module.js";'),
        ("src/module.ts", "export const thing = 1"),
    ])
    assert result["references_unresolved"] == 0


def test_python_relative_module_and_missing_are_separate():
    result = inspect_snapshot_references([
        ("app/routes/auth.py", "from ..services.tokens import issue\nfrom ..services.absent import bad\n"),
        ("app/services/tokens.py", "def issue(): pass"),
    ])
    assert result["references_checked"] == 2
    assert result["references_resolved"] == 1
    assert result["missing_candidates"][0]["specifier"] == "..services.absent"


def test_unresolved_results_are_bounded_and_clear_about_truncation():
    imports = "\n".join(f'import "./missing{i}.svg";' for i in range(10))
    result = inspect_snapshot_references([("src/main.ts", imports)], max_results=3)
    assert result["references_unresolved"] == 10
    assert len(result["missing_candidates"]) == 3
    assert result["results_truncated"] is True


def test_reference_cannot_escape_the_saved_snapshot():
    result = inspect_snapshot_references([
        ("src/main.ts", 'import secret from "../../outside/secret.svg";')
    ])
    assert result["references_checked"] == 0
    assert result["missing_candidates"] == []


def test_binary_assets_without_path_inventory_are_not_falsely_missing():
    result = inspect_snapshot_references([
        ("src/main.ts", 'import image from "./photo.png";\nimport font from "./font.woff2";')
    ])
    assert result["references_checked"] == 0
    assert result["references_unresolved"] == 0
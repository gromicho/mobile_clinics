#!/usr/bin/env python3
"""Dependency-free structural preflight for changed Jupyter notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CONFIG_SUFFIXES = {".json", ".yml", ".yaml", ".toml"}


def changed_notebooks(repo: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return []
    paths: list[Path] = []
    for record in result.stdout.split("\0"):
        if not record:
            continue
        name = record[3:]
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        path = repo / name
        if path.suffix.lower() == ".ipynb" and path.is_file():
            paths.append(path)
    return paths


def registration_mentions(repo: Path, relative: str) -> list[str]:
    mentions: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        rel = path.relative_to(repo)
        if ".git" in rel.parts or any(part in {"build", "dist", ".venv", "venv"} for part in rel.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if relative in text:
            mentions.append(rel.as_posix())
    return mentions


def inspect(path: Path, repo: Path, require_launchers: bool, strict_registration: bool) -> dict[str, object]:
    relative = path.relative_to(repo).as_posix()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": relative, "errors": [f"Invalid notebook JSON: {exc}"], "warnings": []}

    if notebook.get("nbformat") != 4:
        errors.append(f"Expected nbformat 4, found {notebook.get('nbformat')!r}")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return {"path": relative, "errors": errors + ["Notebook cells must be a list"], "warnings": warnings}

    ids: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"Cell {index} is not an object")
            continue
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            errors.append(f"Cell {index} has no stable top-level id")
        elif cell_id in ids:
            errors.append(f"Cell {index} duplicates id {cell_id!r}")
        else:
            ids.add(cell_id)
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                errors.append(f"Code cell {index} has a saved execution count")
            outputs = cell.get("outputs")
            if outputs != []:
                errors.append(f"Code cell {index} has saved output or no explicit outputs list")

    first_source = ""
    if cells:
        source = cells[0].get("source", "") if isinstance(cells[0], dict) else ""
        first_source = "".join(source) if isinstance(source, list) else str(source)
    if require_launchers:
        if "colab.research.google.com" not in first_source:
            errors.append("First cell has no Colab launcher")
        if "mybinder.org" not in first_source:
            errors.append("First cell has no Binder launcher")
        if f"urlpath=tree/{relative}" not in first_source:
            errors.append("Binder launcher does not contain the exact notebook path")
        if relative not in first_source:
            errors.append("Launcher cell does not contain the exact notebook path")

    mentions = registration_mentions(repo, relative)
    if not mentions:
        message = "Notebook path is not referenced by any JSON/YAML/TOML catalogue or manifest"
        (errors if strict_registration else warnings).append(message)
    return {"path": relative, "errors": errors, "warnings": warnings, "registration_mentions": mentions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebook paths; defaults to changed notebooks")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository root")
    parser.add_argument("--strict-registration", action="store_true")
    parser.add_argument("--require-launchers", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    repo = args.repo.resolve()
    paths = [(p if p.is_absolute() else repo / p).resolve() for p in args.notebooks]
    if not paths:
        paths = changed_notebooks(repo)
    if not paths:
        parser.error("No notebook paths supplied and no changed notebooks found")
    outside = [str(p) for p in paths if not p.is_relative_to(repo)]
    if outside:
        parser.error(f"Notebook paths outside repository: {outside}")

    results = [inspect(p, repo, args.require_launchers, args.strict_registration) for p in paths]
    report = {"repository": str(repo), "results": results}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return int(any(result["errors"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

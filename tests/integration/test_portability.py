"""The package must import and run on Windows as well as Unix.

This exists because `core/registry.py` once imported ``fcntl`` for file
locking. fcntl is Unix-only, so on Windows the import blew up and took all three
adapters down with it — the whole project was unusable there, and nothing in the
suite noticed because CI and development were both Linux.

These tests cannot run Windows. What they can do is forbid the *class* of
mistake: no module in ``src/`` may import a platform-specific stdlib module, and
no path may be built by gluing strings with "/".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "onboarding"

# Stdlib modules that do not exist on at least one supported platform.
UNIX_ONLY = {
    "fcntl", "pwd", "grp", "termios", "tty", "pty", "resource",
    "syslog", "posix", "spwd", "crypt", "nis", "ossaudiodev",
}
WINDOWS_ONLY = {"msvcrt", "winreg", "winsound", "_winapi"}
PLATFORM_SPECIFIC = UNIX_ONLY | WINDOWS_ONLY


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_platform_specific_stdlib_imports(path: Path) -> None:
    """A Unix-only import makes the package unusable on Windows, and vice versa."""
    offenders = _imported_modules(ast.parse(path.read_text())) & PLATFORM_SPECIFIC
    assert not offenders, (
        f"{path.relative_to(SRC)} imports {sorted(offenders)}, which does not exist on every "
        "platform. Use a cross-platform library instead — filelock for locking, "
        "pathlib for paths."
    )


def test_the_registry_lock_is_cross_platform() -> None:
    """The specific regression: registry locking must not use fcntl.

    Checked against the import list, not the file text — the module docstring
    legitimately mentions fcntl to explain why it is gone.
    """
    imported = _imported_modules(ast.parse((SRC / "core" / "registry.py").read_text()))
    assert "fcntl" not in imported
    assert "filelock" in imported


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_paths_are_not_built_by_string_concatenation(path: Path) -> None:
    """Hard-coded "/" separators break on Windows. Use pathlib's / operator."""
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or "http" in line or "://" in line:
            continue
        for bad in ('+ "/"', "+ '/'", '"/" +', "'/' +"):
            assert bad not in line, (
                f"{path.relative_to(SRC)}:{line_no} joins a path with a literal '/': {stripped!r}"
            )


def test_every_module_imports_cleanly() -> None:
    """Nothing in the package may fail at import time on this platform."""
    import importlib

    for path in _python_files():
        if path.name == "__init__.py":
            continue
        module = "onboarding." + str(path.relative_to(SRC).with_suffix("")).replace("/", ".")
        importlib.import_module(module)


def test_registry_lock_file_sits_beside_the_csv() -> None:
    from onboarding.core.registry import lock_path, registry_path

    assert lock_path().parent == registry_path().parent
    assert lock_path().name.endswith(".lock")


def test_writing_the_registry_creates_and_releases_the_lock(valid_record) -> None:
    """A completed write must not leave the registry locked."""
    from onboarding.core.registry import append_customer, lock_path, read_all

    append_customer(valid_record, run_id="r1")
    assert len(read_all()) == 1

    # filelock leaves the sentinel file behind but unlocked; a second write
    # must still succeed rather than block until the timeout.
    other = valid_record.model_copy(update={"record_id": "OTHER-1"})
    append_customer(other, run_id="r2")
    assert len(read_all()) == 2
    assert lock_path().parent.exists()

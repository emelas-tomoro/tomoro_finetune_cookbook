from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find the repo root by walking upward until we see expected markers.
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()

    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "data" / "finetuning").exists():
            return p
    raise FileNotFoundError(
        "Could not find repo root (expected pyproject.toml + data/finetuning). "
        f"Started from: {start}"
    )


def ensure_tinker_cookbook_importable(repo_root: Path) -> None:
    """
    The notebooks sometimes rely on a sibling checkout of tinker-cookbook.
    If `tinker_cookbook` can't be imported, try adding a sibling repo to sys.path.
    """
    try:
        import tinker_cookbook  # noqa: F401

        return
    except Exception:
        pass

    import sys

    candidates = [
        repo_root / "tinker-cookbook",
        repo_root.parent / "tinker-cookbook",
    ]
    for c in candidates:
        if (c / "tinker_cookbook" / "__init__.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return

    raise FileNotFoundError(
        "Could not import tinker_cookbook and could not find a sibling 'tinker-cookbook' checkout."
    )



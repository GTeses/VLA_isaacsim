#!/usr/bin/env python3
"""Check local prerequisites for the Zhishu stage-three openpi workflow.

This script is intentionally local-only: it does not download anything.
It reports whether the expected openpi checkpoints and environments are present
before running `compute_norm_stats.py`, `train.py`, or `serve_policy.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zhishu_dualarm_lab.utils.local_paths import resolve_openpi_checkpoint_roots, resolve_openpi_root


@dataclasses.dataclass(frozen=True)
class Args:
    openpi_root: Path = resolve_openpi_root()
    checkpoint_roots: tuple[Path, ...] = resolve_openpi_checkpoint_roots()


def _status_line(name: str, ok: bool, detail: str) -> str:
    prefix = "[OK]" if ok else "[MISSING]"
    return f"{prefix} {name}: {detail}"


def _check_checkpoint(root: Path, checkpoint_name: str) -> list[str]:
    ckpt_dir = root / checkpoint_name
    params_dir = ckpt_dir / "params"
    metadata_path = params_dir / "_METADATA"
    assets_dir = ckpt_dir / "assets"

    lines = [
        _status_line("checkpoint_dir", ckpt_dir.is_dir(), str(ckpt_dir)),
        _status_line("params_metadata", metadata_path.is_file(), str(metadata_path)),
        _status_line("assets_dir", assets_dir.is_dir(), str(assets_dir)),
    ]
    return lines


def _find_checkpoint_root(roots: tuple[Path, ...], checkpoint_name: str) -> Path | None:
    for root in roots:
        if (root / checkpoint_name).exists():
            return root
    return None


def main(args: Args) -> None:
    openpi_python = args.openpi_root / ".venv" / "bin" / "python"

    print("== openpi env ==")
    print(_status_line("openpi_root", args.openpi_root.is_dir(), str(args.openpi_root)))
    print(_status_line("openpi_python", openpi_python.is_file(), str(openpi_python)))

    print("\n== checkpoints ==")
    print("candidate_roots:")
    for root in args.checkpoint_roots:
        print(f"- {root}")

    print("-- pi05_libero --")
    libero_root = _find_checkpoint_root(args.checkpoint_roots, "pi05_libero") or args.checkpoint_roots[0]
    print(f"selected_root: {libero_root}")
    for line in _check_checkpoint(libero_root, "pi05_libero"):
        print(line)

    print("\n-- pi05_base --")
    base_root = _find_checkpoint_root(args.checkpoint_roots, "pi05_base") or args.checkpoint_roots[0]
    print(f"selected_root: {base_root}")
    for line in _check_checkpoint(base_root, "pi05_base"):
        print(line)

    pi05_base_dir = base_root / "pi05_base"
    if not pi05_base_dir.exists():
        print("\nNext action:")
        print("- `pi05_base` is not available in any known local checkpoint root yet.")
        print("- Stage-three training or serving with `pi05_zhishu_dualarm_nohand` should wait until that checkpoint exists.")
        print("- After it is synced locally, rerun this script before `compute_norm_stats.py` / `train.py` / `serve_policy.py`.")
    else:
        print("\nNext action:")
        print(f"- `pi05_base` is ready at: {pi05_base_dir}")
        print("- The remaining blockers are Zhishu dataset conversion, `compute_norm_stats`, and fine-tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi_root", type=Path, default=Args.openpi_root)
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        action="append",
        default=None,
        help="Add a checkpoint root to search. Can be passed multiple times.",
    )
    ns = parser.parse_args()
    checkpoint_roots = tuple(ns.checkpoint_root) if ns.checkpoint_root else Args.checkpoint_roots
    main(Args(openpi_root=ns.openpi_root, checkpoint_roots=checkpoint_roots))

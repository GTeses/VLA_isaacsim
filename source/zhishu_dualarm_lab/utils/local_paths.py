from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source"
PACKAGE_ROOT = SOURCE_ROOT / "zhishu_dualarm_lab"
CONFIG_DIR = REPO_ROOT / "config"
LOCAL_PATHS_FILE = CONFIG_DIR / "local_paths.json"


def _load_local_config() -> dict[str, Any]:
    if not LOCAL_PATHS_FILE.is_file():
        return {}
    try:
        data = json.loads(LOCAL_PATHS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in local path config: {LOCAL_PATHS_FILE}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Local path config must contain a JSON object: {LOCAL_PATHS_FILE}")
    return data


_LOCAL_CONFIG = _load_local_config()


def _unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)


def _path_from_config(key: str) -> Path | None:
    value = _LOCAL_CONFIG.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Config key '{key}' in {LOCAL_PATHS_FILE} must be a non-empty string.")
    return Path(value).expanduser()


def _paths_from_config(key: str) -> tuple[Path, ...]:
    value = _LOCAL_CONFIG.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise RuntimeError(f"Config key '{key}' in {LOCAL_PATHS_FILE} must be a list of non-empty strings.")
    return tuple(Path(item).expanduser() for item in value)


def _split_env_paths(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    return tuple(Path(part).expanduser() for part in value.split(os.pathsep) if part.strip())


def resolve_robot_urdf_path() -> Path:
    env_path = os.environ.get("ZHISHU_ROBOT_URDF")
    config_path = _path_from_config("robot_urdf_path")
    default_candidates = [
        REPO_ROOT.parent
        / "zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf",
        REPO_ROOT
        / "third_party/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf",
    ]
    candidates = [Path(env_path).expanduser()] if env_path else []
    if config_path is not None:
        candidates.append(config_path)
    candidates.extend(default_candidates)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_openpi_root() -> Path:
    env_path = os.environ.get("OPENPI_ROOT")
    config_path = _path_from_config("openpi_root")
    default_path = REPO_ROOT.parent / "openpi"
    return Path(env_path).expanduser() if env_path else (config_path or default_path)


def resolve_openpi_checkpoint_roots() -> tuple[Path, ...]:
    env_roots = _split_env_paths(os.environ.get("OPENPI_CHECKPOINT_ROOTS"))
    legacy_env_root = os.environ.get("OPENPI_LOCAL_CHECKPOINT_ROOT")
    config_roots = _paths_from_config("openpi_checkpoint_roots")
    default_roots = (
        resolve_openpi_root() / "openpi-assets" / "checkpoints",
        Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints",
    )
    roots: list[Path] = list(env_roots)
    if legacy_env_root:
        roots.append(Path(legacy_env_root).expanduser())
    roots.extend(config_roots)
    roots.extend(default_roots)
    return _unique_paths(roots)

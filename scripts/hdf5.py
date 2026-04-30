#!/usr/bin/env python3
"""Inspect a clean closing-in HDF5 dataset and print its structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


def _normalize_path(path: Path) -> Path:
    return path if path.suffix == ".hdf5" else path.with_suffix(".hdf5")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _format_scalar(value.item())
    return repr(value)


def _print_attrs(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    if not obj.attrs:
        return
    print(f"{indent}attrs:")
    for key in sorted(obj.attrs.keys()):
        print(f"{indent}  - {key}: {_format_scalar(obj.attrs[key])}")


def _describe_dataset(name: str, dataset: h5py.Dataset, indent: str, preview_arrays: bool) -> None:
    shape = tuple(dataset.shape)
    dtype = str(dataset.dtype)
    print(f"{indent}- {name}: dataset shape={shape} dtype={dtype}")
    _print_attrs(dataset, indent + "  ")

    if not preview_arrays:
        return

    try:
        value = dataset[()]
    except Exception as exc:
        print(f"{indent}  preview: <failed to read: {exc}>")
        return

    if isinstance(value, bytes):
        print(f"{indent}  preview: {value.decode('utf-8', errors='replace')!r}")
        return
    if np.isscalar(value):
        print(f"{indent}  preview: {value!r}")
        return
    arr = np.asarray(value)
    if arr.size == 0:
        print(f"{indent}  preview: []")
        return
    if arr.ndim == 1 and arr.size <= 16:
        print(f"{indent}  preview: {np.round(arr.astype(np.float64), 4).tolist()}")
        return
    print(f"{indent}  preview: first element shape={arr[0].shape if arr.ndim > 0 else ()}")


def _walk_group(group: h5py.Group, indent: str, preview_arrays: bool) -> None:
    _print_attrs(group, indent)
    for key in sorted(group.keys()):
        value = group[key]
        if isinstance(value, h5py.Group):
            print(f"{indent}- {key}/")
            _walk_group(value, indent + "  ", preview_arrays)
        else:
            _describe_dataset(key, value, indent, preview_arrays)


def _print_episode_summary(data_group: h5py.Group) -> None:
    episode_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
    print()
    print(f"[INFO] episodes={len(episode_names)} total_samples={int(data_group.attrs.get('total', 0))}")
    for episode_name in episode_names[:10]:
        episode_group = data_group[episode_name]
        prompt = episode_group.attrs.get("prompt", "")
        steps = int(episode_group.attrs.get("num_samples", 0))
        success = bool(episode_group.attrs.get("success", False))
        print(f"  - {episode_name}: steps={steps} success={success} prompt={prompt!r}")
    if len(episode_names) > 10:
        print(f"  ... ({len(episode_names) - 10} more episodes)")


def _save_episode_images(
    data_group: h5py.Group,
    *,
    episode_indices: list[int],
    step_idx: int | None,
    save_images_dir: Path,
) -> None:
    episode_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
    save_images_dir.mkdir(parents=True, exist_ok=True)

    image_keys = (
        ("external", "external_image"),
        ("left_wrist", "left_wrist_image"),
        ("right_wrist", "right_wrist_image"),
    )

    for episode_index in episode_indices:
        if episode_index >= len(episode_names):
            raise IndexError(f"Episode index {episode_index} is out of range for {len(episode_names)} episodes.")

        episode_name = episode_names[episode_index]
        episode_group = data_group[episode_name]
        obs_group = episode_group["observation"]
        num_steps = int(obs_group["state"].shape[0])
        step_indices = [step_idx] if step_idx is not None else list(range(num_steps))

        if step_idx is not None and (step_idx < 0 or step_idx >= num_steps):
            raise IndexError(f"Step index {step_idx} is out of range for {episode_name} with {num_steps} steps.")

        episode_dir = save_images_dir / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] saving images for {episode_name} steps={step_indices if step_idx is not None else f'0..{num_steps - 1}'}")
        for current_step in step_indices:
            step_dir = episode_dir / f"step_{current_step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for image_prefix, dataset_name in image_keys:
                image = np.asarray(obs_group[dataset_name][current_step], dtype=np.uint8)
                image_path = step_dir / f"{image_prefix}.png"
                Image.fromarray(image).save(image_path)
                print(f"  - {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_file", type=Path, help="Absolute or relative path to the HDF5 file.")
    parser.add_argument("--root_group", default="data", help="Root group to inspect. Default: data")
    parser.add_argument("--preview_arrays", action="store_true", help="Print small value previews for datasets.")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=[], help="Episode indices whose images should be exported.")
    parser.add_argument("--step_idx", type=int, help="Optional single step index to export. If omitted, export all steps in the selected episodes.")
    parser.add_argument("--save_images_dir", type=Path, help="Directory for exported PNG images.")
    args = parser.parse_args()

    dataset_path = _normalize_path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {dataset_path}")

    print(f"[INFO] file={dataset_path}")
    with h5py.File(dataset_path, "r") as handle:
        print(f"[INFO] top-level groups={list(handle.keys())}")
        if args.root_group not in handle:
            raise KeyError(f"Root group {args.root_group!r} not found in {dataset_path}")
        root_group = handle[args.root_group]
        print(f"[INFO] inspecting /{args.root_group}")
        _walk_group(root_group, indent="  ", preview_arrays=args.preview_arrays)
        _print_episode_summary(root_group)
        if args.episode_indices:
            if args.save_images_dir is None:
                raise ValueError("--save_images_dir is required when --episode_indices is used.")
            _save_episode_images(
                root_group,
                episode_indices=args.episode_indices,
                step_idx=args.step_idx,
                save_images_dir=args.save_images_dir,
            )


if __name__ == "__main__":
    main()

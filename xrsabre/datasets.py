"""Discovery of reduced XRS exports below a configured workspace root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReducedDataset:
    dataset_id: str
    directory: Path
    data_file: Path
    roi_file: Path
    scan_name: str


def _scan_root(path: Path) -> str:
    name = path.name
    for suffix in ("_all_data.txt", "_data.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_reduced_datasets(root: Path) -> tuple[ReducedDataset, ...]:
    """Recursively find valid data/ROI export pairs, preferring all-data files."""
    root = root.resolve()
    if not root.is_dir():
        return ()
    selected: dict[tuple[Path, str], Path] = {}
    for pattern in ("*_data.txt", "*_all_data.txt"):
        for data_file in sorted(root.rglob(pattern)):
            scan_name = _scan_root(data_file)
            roi_file = data_file.parent / f"{scan_name}_rois.txt"
            if not roi_file.is_file():
                continue
            key = (data_file.parent, scan_name)
            current = selected.get(key)
            if current is None or data_file.name.endswith("_all_data.txt"):
                selected[key] = data_file

    records: list[ReducedDataset] = []
    directory_counts: dict[Path, int] = {}
    for directory, _scan_name in selected:
        directory_counts[directory] = directory_counts.get(directory, 0) + 1
    for (directory, scan_name), data_file in selected.items():
        relative = directory.relative_to(root).as_posix()
        base_id = relative if relative != "." else scan_name
        dataset_id = (
            f"{base_id}/{scan_name}"
            if relative != "." and directory_counts[directory] > 1
            else base_id
        )
        records.append(ReducedDataset(
            dataset_id=dataset_id,
            directory=directory,
            data_file=data_file,
            roi_file=directory / f"{scan_name}_rois.txt",
            scan_name=scan_name,
        ))
    return tuple(sorted(records, key=lambda item: item.dataset_id.casefold()))


def select_reduced_dataset(root: Path, dataset_id: str) -> ReducedDataset:
    records = discover_reduced_datasets(root)
    matches = [record for record in records if record.dataset_id == dataset_id]
    if not matches:
        available = ", ".join(record.dataset_id for record in records) or "<none>"
        raise KeyError(f"Unknown dataset {dataset_id!r}; available: {available}")
    return matches[0]


__all__ = ["ReducedDataset", "discover_reduced_datasets", "select_reduced_dataset"]

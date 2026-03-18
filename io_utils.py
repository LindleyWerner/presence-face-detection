from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

from config import DEFAULT_SETTINGS, Settings


PERSON_PATTERN = re.compile(r"^person_(\d{4})$")


@dataclass
class ProjectPaths:
    root: Path
    project_name: str
    project_dir: Path
    people_dir: Path
    photos_dir: Path
    attendance_csv: Path
    log_file: Path


def build_project_paths(root: Path | str, project_name: str, settings: Settings = DEFAULT_SETTINGS) -> ProjectPaths:
    root_path = Path(root).expanduser().resolve()
    project_dir = root_path / project_name
    return ProjectPaths(
        root=root_path,
        project_name=project_name,
        project_dir=project_dir,
        people_dir=project_dir / settings.people_dirname,
        photos_dir=project_dir / settings.photos_dirname,
        attendance_csv=project_dir / settings.attendance_filename,
        log_file=project_dir / settings.log_filename,
    )


def ensure_project_structure(root: Path | str, project_name: str, settings: Settings = DEFAULT_SETTINGS) -> ProjectPaths:
    paths = build_project_paths(root, project_name, settings)
    paths.project_dir.mkdir(parents=True, exist_ok=True)
    paths.people_dir.mkdir(parents=True, exist_ok=True)
    paths.photos_dir.mkdir(parents=True, exist_ok=True)
    paths.log_file.touch(exist_ok=True)
    return paths


def ensure_photo_date_dir(paths: ProjectPaths, date_str: str) -> Path:
    date_dir = paths.photos_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def list_image_files(directory: Path, settings: Settings = DEFAULT_SETTINGS) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in settings.allowed_image_suffixes]
    )


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default.copy()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def iter_person_dirs(paths: ProjectPaths) -> list[Path]:
    if not paths.people_dir.exists():
        return []
    return sorted([path for path in paths.people_dir.iterdir() if path.is_dir()])


def next_person_id(paths: ProjectPaths) -> str:
    max_id = 0
    for person_dir in iter_person_dirs(paths):
        match = PERSON_PATTERN.match(person_dir.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"person_{max_id + 1:04d}"


def append_log(paths: ProjectPaths, message: str) -> None:
    with paths.log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")

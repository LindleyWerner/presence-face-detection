from __future__ import annotations

from pathlib import Path

from config import DEFAULT_SETTINGS, Settings
from io_utils import build_project_paths, ensure_project_structure


class ProjectIndex:
    def __init__(self, root: Path | str, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.root = Path(root).expanduser().resolve()
        self.settings = settings
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[str]:
        return sorted([path.name for path in self.root.iterdir() if path.is_dir()])

    def create_project(self, project_name: str) -> None:
        ensure_project_structure(self.root, project_name, self.settings)

    def project_exists(self, project_name: str) -> bool:
        return build_project_paths(self.root, project_name, self.settings).project_dir.exists()

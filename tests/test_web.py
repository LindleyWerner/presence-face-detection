from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import create_app
from attendance import AttendanceManager
from io_utils import ensure_project_structure, save_json


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.app = create_app(str(self.root))
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_home_page_loads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Face Detection Attendance", response.data)

    def test_create_project_redirects_to_project_page(self) -> None:
        response = self.client.post("/projects", data={"project_name": "Project_A"})
        self.assertEqual(response.status_code, 302)
        self.assertTrue((self.root / "Project_A").exists())

    def test_project_page_hides_folder_label(self) -> None:
        self.client.post("/projects", data={"project_name": "Project_A"})
        response = self.client.get("/projects/Project_A")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Folder:", response.data)

    def test_delete_person_route_removes_person_from_project(self) -> None:
        paths = ensure_project_structure(self.root, "Project_A")
        person_dir = paths.people_dir / "person_0001"
        person_dir.mkdir(parents=True, exist_ok=True)
        save_json(person_dir / "metadata.json", {"display_name": "person_0001"})
        AttendanceManager(paths).mark_present("person_0001", "2025-08-15")

        response = self.client.post("/projects/Project_A/delete", data={"person_name": "person_0001"})

        self.assertEqual(response.status_code, 302)
        self.assertFalse(person_dir.exists())
        frame = AttendanceManager(paths).load()
        self.assertEqual(frame.shape[0], 0)

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import create_app


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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from attendance import AttendanceManager
from io_utils import ensure_project_structure


class AttendanceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.paths = ensure_project_structure(self.root, "Project_A")
        self.manager = AttendanceManager(self.paths)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mark_present_creates_row_and_date_column(self) -> None:
        self.manager.mark_present("person_0001", "2025-08-15")

        frame = self.manager.load()
        self.assertEqual(list(frame.columns), ["Name", "2025-08-15"])
        self.assertEqual(frame.iloc[0]["Name"], "person_0001")
        self.assertEqual(int(frame.iloc[0]["2025-08-15"]), 1)

    def test_rename_person_updates_attendance_row(self) -> None:
        self.manager.mark_present("person_0001", "2025-08-15")

        self.manager.rename_person("person_0001", "alice")
        frame = self.manager.load()

        self.assertIn("alice", frame["Name"].values)
        self.assertNotIn("person_0001", frame["Name"].values)

    def test_merge_people_uses_or_logic_for_dates(self) -> None:
        self.manager.mark_present("person_0001", "2025-08-15")
        self.manager.mark_present("person_0002", "2025-08-16")

        self.manager.merge_people("person_0001", "person_0002")
        frame = self.manager.load()

        self.assertEqual(frame.shape[0], 1)
        row = frame.iloc[0]
        self.assertEqual(row["Name"], "person_0001")
        self.assertEqual(int(row["2025-08-15"]), 1)
        self.assertEqual(int(row["2025-08-16"]), 1)


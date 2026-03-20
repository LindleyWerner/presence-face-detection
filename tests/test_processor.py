from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from attendance import AttendanceManager
from io_utils import ensure_project_structure, load_json, save_json
from processor import Processor


class ProcessorIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project = "Project_A"
        self.paths = ensure_project_structure(self.root, self.project)
        self.processor = Processor()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_person(self, name: str, embedding_rows: int = 1) -> Path:
        person_dir = self.paths.people_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.ones((embedding_rows, 4), dtype=np.float32)
        np.save(person_dir / self.processor.settings.embeddings_filename, embeddings)
        save_json(
            person_dir / self.processor.settings.metadata_filename,
            {
                "display_name": name,
                "canonical_score": float(embedding_rows),
                "canonical_image": self.processor.settings.canonical_filename,
            },
        )
        (person_dir / self.processor.settings.canonical_filename).write_bytes(b"fake-image")
        (person_dir / f"{name}_face_01.jpg").write_bytes(b"fake-crop")
        return person_dir

    def test_rename_person_moves_folder_and_updates_csv(self) -> None:
        self._create_person("person_0001")
        AttendanceManager(self.paths).mark_present("person_0001", "2025-08-15")

        self.processor.rename_person(self.root, self.project, "person_0001", "alice")

        self.assertFalse((self.paths.people_dir / "person_0001").exists())
        self.assertTrue((self.paths.people_dir / "alice").exists())
        metadata = load_json((self.paths.people_dir / "alice") / self.processor.settings.metadata_filename)
        self.assertEqual(metadata["display_name"], "alice")
        frame = AttendanceManager(self.paths).load()
        self.assertIn("alice", frame["Name"].values)

    def test_merge_people_combines_embeddings_and_attendance(self) -> None:
        target_dir = self._create_person("person_0001", embedding_rows=2)
        self._create_person("person_0002", embedding_rows=3)
        attendance = AttendanceManager(self.paths)
        attendance.mark_present("person_0001", "2025-08-15")
        attendance.mark_present("person_0002", "2025-08-16")

        self.processor.merge_people(self.root, self.project, "person_0001", "person_0002")

        merged_embeddings = np.load(target_dir / self.processor.settings.embeddings_filename)
        self.assertEqual(merged_embeddings.shape[0], 5)
        self.assertFalse((self.paths.people_dir / "person_0002").exists())
        frame = attendance.load()
        self.assertEqual(frame.shape[0], 1)
        row = frame.iloc[0]
        self.assertEqual(int(row["2025-08-15"]), 1)
        self.assertEqual(int(row["2025-08-16"]), 1)

    def test_delete_person_removes_folder_and_attendance(self) -> None:
        self._create_person("person_0001")
        attendance = AttendanceManager(self.paths)
        attendance.mark_present("person_0001", "2025-08-15")

        self.processor.delete_person(self.root, self.project, "person_0001")

        self.assertFalse((self.paths.people_dir / "person_0001").exists())
        frame = attendance.load()
        self.assertEqual(frame.shape[0], 0)

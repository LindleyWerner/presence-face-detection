from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np

from attendance import AttendanceManager
from config import DEFAULT_SETTINGS, Settings
from face_engine import DetectionResult, FaceEngine, StoredPerson
from io_utils import (
    append_log,
    build_project_paths,
    ensure_project_structure,
    list_image_files,
    load_json,
    next_person_id,
    save_json,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class ProcessingSummary:
    project: str
    date_str: str
    images_processed: int = 0
    detections_kept: int = 0
    matches: int = 0
    new_people: int = 0
    skipped_small_faces: int = 0


class Processor:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings
        self.face_engine = FaceEngine(settings)

    def process(self, root: Path | str, project_name: str, date_str: str) -> ProcessingSummary:
        paths = ensure_project_structure(root, project_name, self.settings)
        image_dir = paths.photos_dir / date_str
        attendance = AttendanceManager(paths)
        people = self.face_engine.load_people(paths)
        summary = ProcessingSummary(project=project_name, date_str=date_str)

        for image_path in list_image_files(image_dir, self.settings):
            summary.images_processed += 1
            for index, detection in enumerate(self.face_engine.detect_faces(image_path), start=1):
                if min(detection.box[2], detection.box[3]) < self.settings.min_face_size:
                    summary.skipped_small_faces += 1
                    append_log(
                        paths,
                        f"SKIP_SMALL {image_path.name} | backend={detection.backend} | conf={detection.confidence:.4f} "
                        f"| box={detection.box}",
                    )
                    continue

                try:
                    embedding = self.face_engine.extract_embedding(detection.face_rgb)
                except Exception as exc:
                    append_log(
                        paths,
                        f"EMBED_FAIL {image_path.name} | backend={detection.backend} | conf={detection.confidence:.4f} "
                        f"| error={exc}",
                    )
                    continue

                matched_person, distance = self.face_engine.match_person(embedding, people)
                event = "MATCH"
                if matched_person is None:
                    person_id = next_person_id(paths)
                    person_folder = paths.people_dir / person_id
                    person_folder.mkdir(parents=True, exist_ok=True)
                    matched_person = StoredPerson(
                        person_id=person_id,
                        folder=person_folder,
                        embeddings=np.empty((0, embedding.shape[0]), dtype=np.float32),
                        metadata={
                            "display_name": person_id,
                            "canonical_image": self.settings.canonical_filename,
                            "canonical_score": 0.0,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "last_seen": date_str,
                        },
                    )
                    people.append(matched_person)
                    summary.new_people += 1
                    event = "NEW"
                else:
                    summary.matches += 1

                self._store_face_record(matched_person, detection, embedding, image_path, index, date_str)
                people = self.face_engine.load_people(paths)
                attendance.mark_present(matched_person.person_id, date_str)
                summary.detections_kept += 1
                distance_text = "NA" if distance is None else f"{distance:.4f}"
                append_log(
                    paths,
                    f"{event} {matched_person.person_id} | {image_path.name} | dist={distance_text} "
                    f"| conf={detection.confidence:.4f} | backend={detection.backend} | box={detection.box}",
                )

        append_log(
            paths,
            f"SUMMARY date={date_str} | images={summary.images_processed} | detections={summary.detections_kept} "
            f"| matches={summary.matches} | new={summary.new_people} | skipped_small={summary.skipped_small_faces}",
        )
        return summary

    def rename_person(self, root: Path | str, project_name: str, current_name: str, new_name: str) -> None:
        paths = build_project_paths(root, project_name, self.settings)
        source_dir = paths.people_dir / current_name
        target_dir = paths.people_dir / new_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Person folder not found: {current_name}")
        if target_dir.exists():
            raise FileExistsError(f"Target person folder already exists: {new_name}")
        source_dir.rename(target_dir)
        metadata_path = target_dir / self.settings.metadata_filename
        metadata = load_json(metadata_path, default={"display_name": new_name})
        metadata["display_name"] = new_name
        save_json(metadata_path, metadata)
        AttendanceManager(paths).rename_person(current_name, new_name)
        append_log(paths, f"RENAME {current_name} -> {new_name}")

    def merge_people(self, root: Path | str, project_name: str, target_name: str, source_name: str) -> None:
        paths = build_project_paths(root, project_name, self.settings)
        target_dir = paths.people_dir / target_name
        source_dir = paths.people_dir / source_name
        if not target_dir.exists():
            raise FileNotFoundError(f"Target person folder not found: {target_name}")
        if not source_dir.exists():
            raise FileNotFoundError(f"Source person folder not found: {source_name}")
        if target_name == source_name:
            raise ValueError("Cannot merge a person into the same person.")

        target_embeddings = self._load_embeddings(target_dir)
        source_embeddings = self._load_embeddings(source_dir)
        arrays = [array for array in [target_embeddings, source_embeddings] if array.size > 0]
        if arrays:
            combined = np.vstack(arrays)
            if combined.shape[0] > self.settings.max_embeddings_per_person:
                combined = combined[-self.settings.max_embeddings_per_person :]
            np.save(target_dir / self.settings.embeddings_filename, combined.astype(np.float32))

        self._move_face_crops(source_dir, target_dir)
        self._merge_canonical(target_dir, source_dir)
        AttendanceManager(paths).merge_people(target_name, source_name)
        shutil.rmtree(source_dir)
        append_log(paths, f"MERGE {source_name} -> {target_name}")

    def delete_person(self, root: Path | str, project_name: str, person_name: str) -> None:
        paths = build_project_paths(root, project_name, self.settings)
        person_dir = paths.people_dir / person_name
        if not person_dir.exists():
            raise FileNotFoundError(f"Person folder not found: {person_name}")
        shutil.rmtree(person_dir)
        AttendanceManager(paths).delete_person(person_name)
        append_log(paths, f"DELETE {person_name}")

    def _store_face_record(
        self,
        person: StoredPerson,
        detection: DetectionResult,
        embedding: np.ndarray,
        image_path: Path,
        detection_index: int,
        date_str: str,
    ) -> None:
        person.folder.mkdir(parents=True, exist_ok=True)
        crop_path = person.folder / f"{image_path.stem}_face_{detection_index:02d}.jpg"
        self._save_image(crop_path, detection.face_rgb)

        stored_embeddings = self._load_embeddings(person.folder)
        combined = embedding.reshape(1, -1) if stored_embeddings.size == 0 else np.vstack([stored_embeddings, embedding.reshape(1, -1)])
        if combined.shape[0] > self.settings.max_embeddings_per_person:
            combined = combined[-self.settings.max_embeddings_per_person :]
        np.save(person.folder / self.settings.embeddings_filename, combined.astype(np.float32))

        metadata_path = person.folder / self.settings.metadata_filename
        metadata = load_json(metadata_path, default={"display_name": person.person_id})
        metadata.setdefault("display_name", person.person_id)
        metadata["last_seen"] = date_str
        canonical_score = detection.area * max(detection.confidence, 0.01)
        if canonical_score >= float(metadata.get("canonical_score", 0.0)):
            metadata["canonical_score"] = canonical_score
            metadata["canonical_image"] = self.settings.canonical_filename
            self._save_image(person.folder / self.settings.canonical_filename, detection.face_rgb)
        save_json(metadata_path, metadata)

    def _load_embeddings(self, person_dir: Path) -> np.ndarray:
        path = person_dir / self.settings.embeddings_filename
        if not path.exists():
            return np.empty((0, 0), dtype=np.float32)
        embeddings = np.load(path)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32, copy=False)

    def _move_face_crops(self, source_dir: Path, target_dir: Path) -> None:
        for file_path in source_dir.iterdir():
            if file_path.name in {
                self.settings.embeddings_filename,
                self.settings.metadata_filename,
                self.settings.canonical_filename,
            }:
                continue
            destination = target_dir / file_path.name
            if destination.exists():
                destination = target_dir / f"{file_path.stem}_merged{file_path.suffix}"
            shutil.move(str(file_path), str(destination))

    def _merge_canonical(self, target_dir: Path, source_dir: Path) -> None:
        target_meta_path = target_dir / self.settings.metadata_filename
        source_meta_path = source_dir / self.settings.metadata_filename
        target_meta = load_json(target_meta_path, default={"display_name": target_dir.name, "canonical_score": 0.0})
        source_meta = load_json(source_meta_path, default={"display_name": source_dir.name, "canonical_score": 0.0})
        if float(source_meta.get("canonical_score", 0.0)) > float(target_meta.get("canonical_score", 0.0)):
            source_canonical = source_dir / self.settings.canonical_filename
            if source_canonical.exists():
                shutil.copy2(source_canonical, target_dir / self.settings.canonical_filename)
                target_meta["canonical_score"] = source_meta.get("canonical_score", 0.0)
        save_json(target_meta_path, target_meta)

    def _save_image(self, destination: Path, image_rgb: np.ndarray) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to save cropped faces.")
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(destination), image_bgr)

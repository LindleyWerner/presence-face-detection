from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import DEFAULT_SETTINGS, Settings
from io_utils import ProjectPaths, iter_person_dirs, load_json

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = exc
else:
    DEEPFACE_IMPORT_ERROR = None


@dataclass
class DetectionResult:
    backend: str
    confidence: float
    box: tuple[int, int, int, int]
    face_rgb: np.ndarray

    @property
    def area(self) -> int:
        _, _, width, height = self.box
        return max(width, 0) * max(height, 0)


@dataclass
class StoredPerson:
    person_id: str
    folder: Path
    embeddings: np.ndarray
    metadata: dict[str, Any]


class FaceEngine:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def validate_runtime(self) -> None:
        missing: list[str] = []
        if DeepFace is None:
            missing.append("deepface")
        if cv2 is None:
            missing.append("opencv-python")
        if missing:
            details: list[str] = []
            if DEEPFACE_IMPORT_ERROR is not None:
                resolution_hint = ""
                if (
                    isinstance(DEEPFACE_IMPORT_ERROR, ValueError)
                    and "requires tf-keras package" in str(DEEPFACE_IMPORT_ERROR)
                ):
                    resolution_hint = " Install it with: pip install tf-keras"
                details.append(
                    "DeepFace import failed with "
                    f"{type(DEEPFACE_IMPORT_ERROR).__name__}: {DEEPFACE_IMPORT_ERROR}.{resolution_hint}"
                )
            detail_text = f" Details: {' | '.join(details)}" if details else ""
            raise RuntimeError(f"Missing required runtime dependencies: {', '.join(missing)}.{detail_text}")

    def detect_faces(self, image_path: Path) -> list[DetectionResult]:
        self.validate_runtime()
        detections: list[DetectionResult] = []
        for backend in self.settings.detectors:
            try:
                extracted = DeepFace.extract_faces(
                    img_path=str(image_path),
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                    expand_percentage=self.settings.expand_percentage,
                )
            except Exception:
                continue

            for item in extracted:
                face = item.get("face")
                area = item.get("facial_area") or {}
                confidence = float(item.get("confidence") or 0.0)
                if face is None:
                    continue
                detections.append(
                    DetectionResult(
                        backend=backend,
                        confidence=confidence,
                        box=(
                            int(area.get("x", 0)),
                            int(area.get("y", 0)),
                            int(area.get("w", 0)),
                            int(area.get("h", 0)),
                        ),
                        face_rgb=self._normalize_face(face),
                    )
                )
        return self._deduplicate_detections(detections)

    def extract_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        self.validate_runtime()
        representations = DeepFace.represent(
            img_path=face_rgb,
            model_name=self.settings.recognition_model,
            detector_backend="skip",
            enforce_detection=False,
            normalization="base",
        )
        if not representations:
            raise ValueError("DeepFace returned no embeddings.")
        embedding = np.asarray(representations[0]["embedding"], dtype=np.float32)
        return self._normalize_embedding(embedding)

    def load_people(self, paths: ProjectPaths) -> list[StoredPerson]:
        people: list[StoredPerson] = []
        for person_dir in iter_person_dirs(paths):
            embeddings_path = person_dir / self.settings.embeddings_filename
            metadata_path = person_dir / self.settings.metadata_filename
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)
            people.append(
                StoredPerson(
                    person_id=person_dir.name,
                    folder=person_dir,
                    embeddings=embeddings.astype(np.float32, copy=False),
                    metadata=load_json(metadata_path, default={"display_name": person_dir.name}),
                )
            )
        return people

    def match_person(self, embedding: np.ndarray, people: list[StoredPerson]) -> tuple[StoredPerson | None, float | None]:
        best_person: StoredPerson | None = None
        best_distance: float | None = None
        for person in people:
            if person.embeddings.size == 0:
                continue
            distances = 1.0 - np.dot(person.embeddings, embedding)
            person_distance = float(np.min(distances))
            if best_distance is None or person_distance < best_distance:
                best_distance = person_distance
                best_person = person
        if best_distance is not None and best_distance <= self.settings.cosine_threshold:
            return best_person, best_distance
        return None, best_distance

    def _normalize_face(self, face: np.ndarray) -> np.ndarray:
        face_array = np.asarray(face)
        if face_array.dtype != np.uint8:
            scale = 255 if face_array.max() <= 1.0 else 1
            face_array = np.clip(face_array * scale, 0, 255).astype(np.uint8)
        if face_array.ndim == 2:
            face_array = np.stack([face_array] * 3, axis=-1)
        return face_array

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _deduplicate_detections(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        ordered = sorted(detections, key=lambda item: (item.confidence, item.area), reverse=True)
        kept: list[DetectionResult] = []
        for candidate in ordered:
            if any(self._iou(candidate.box, other.box) >= 0.6 for other in kept):
                continue
            kept.append(candidate)
        return kept

    def _iou(self, first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        intersection = inter_w * inter_h
        if intersection == 0:
            return 0.0
        union = (w1 * h1) + (w2 * h2) - intersection
        return intersection / union if union else 0.0

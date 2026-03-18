from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys


def get_runtime_base() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


@dataclass
class Settings:
    detectors: list[str] = field(default_factory=lambda: ["retinaface", "mtcnn"])
    recognition_model: str = "ArcFace"
    cosine_threshold: float = 0.5
    min_face_size: int = 20
    expand_percentage: int = 10
    max_embeddings_per_person: int = 200
    attendance_filename: str = "project_attendance.csv"
    log_filename: str = "processing.log"
    people_dirname: str = "people"
    photos_dirname: str = "photos"
    metadata_filename: str = "metadata.json"
    embeddings_filename: str = "embeddings.npy"
    canonical_filename: str = "canonical.jpg"
    allowed_image_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


BASE_PATH = get_runtime_base()
DEFAULT_SETTINGS = Settings()
DETECTORS = DEFAULT_SETTINGS.detectors
RECOGNITION_MODEL = DEFAULT_SETTINGS.recognition_model
COSINE_THRESHOLD = DEFAULT_SETTINGS.cosine_threshold
MIN_FACE_SIZE = DEFAULT_SETTINGS.min_face_size

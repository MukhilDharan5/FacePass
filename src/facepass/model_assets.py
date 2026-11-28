"""Pinned development asset; verification is local and performs no downloads."""

import hashlib
import os
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAX_MODEL_BYTES = 8_000_000


def default_model_path():
    """Allow an installed launcher to select its model without changing cwd."""
    return os.environ.get("FACEPASS_MODEL") or "models/face_landmarker.task"


def load_verified_model(path):
    with Path(path).open("rb") as source:
        data = source.read(MAX_MODEL_BYTES + 1)
    if len(data) > MAX_MODEL_BYTES or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("model integrity verification failed")
    return data

"""Replaceable local face-landmark provider. No downloads or camera access here."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .model_assets import load_verified_model


@dataclass(frozen=True)
class FaceObservation:
    # Biometric geometry is transient engine input; do not serialize it to logs.
    faces: tuple[np.ndarray, ...]

    def validate(self):
        if not isinstance(self.faces, tuple) or len(self.faces) > 4:
            raise ValueError("malformed face collection")
        for face in self.faces:
            if not isinstance(face, np.ndarray) or face.shape != (478, 3):
                raise ValueError("malformed landmarks")
            if not np.issubdtype(face.dtype, np.floating) or not np.isfinite(face).all():
                raise ValueError("non-finite landmarks")
            if np.max(np.abs(face)) > 10:
                raise ValueError("landmarks outside plausible range")


class MediaPipeLandmarker:
    def __init__(self, model_path: str | Path):
        model_bytes = load_verified_model(model_path)
        # Lazy import keeps the API importable without model initialization.
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._model = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_buffer=model_bytes, delegate=python.BaseOptions.Delegate.CPU,
                ),
                # Redetect all faces on every frame, including new nearby faces.
                # Separate FaceTracker owns continuity; never crop inference to
                # one tracked face and accidentally ignore a second person.
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def detect(self, frame: np.ndarray, timestamp: float) -> FaceObservation:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._model.detect(image)
        faces = tuple(np.array([(p.x, p.y, p.z) for p in face], dtype=np.float64)
                      for face in result.face_landmarks)
        observation = FaceObservation(faces)
        observation.validate()
        return observation

    def close(self):
        self._model.close()

"""Pinned, local YuNet/SFace identity models. No network access at runtime."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .quality import prepare_landmark_frame

AUTH_MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232589),
    "face_recognition_sface_2021dec.onnx": (
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38696353),
}
AUTH_MODEL_DIR = Path("/opt/facepass-auth/current/models")


def verify_auth_model(path):
    path = Path(path)
    digest, size = AUTH_MODELS[path.name]
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("identity model integrity failure")
    return path


def normalized_feature(value):
    value = np.asarray(value)
    if value.size != 128 or value.dtype.kind not in "fi" or not np.isfinite(value).all():
        raise ValueError("invalid identity feature")
    result = value.astype(np.float64).reshape(128)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("empty identity feature")
    return result / norm


def identity_similarity(feature, templates):
    feature = normalized_feature(feature)
    if not 3 <= len(templates) <= 12:
        raise ValueError("invalid template count")
    scores = sorted(float(np.clip(np.dot(feature, normalized_feature(row)), -1, 1))
                    for row in templates)
    # Require support from three enrollment views, rather than one lucky match.
    return float(np.mean(scores[-3:]))


class IdentityModel:
    def __init__(self, directory=AUTH_MODEL_DIR):
        paths = [verify_auth_model(Path(directory) / name) for name in AUTH_MODELS]
        self.detector = cv2.FaceDetectorYN.create(str(paths[0]), "", (640, 480), .8, .3, 100)
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths[1]), "")

    def feature(self, frame, landmarks):
        image, _ = prepare_landmark_frame(frame)
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        if (faces is None or faces.shape != (1, 15) or not np.isfinite(faces).all()
                or faces[0, 14] < .8 or (faces[0, 2:4] <= 0).any()):
            return None  # A second detector must also agree on a single face.
        box = faces[0, :4]
        xy = landmarks[:468, :2] * [image.shape[1], image.shape[0]]
        low, high = xy.min(axis=0), xy.max(axis=0)
        intersection = np.maximum(0, np.minimum(high, box[:2] + box[2:]) - np.maximum(low, box[:2]))
        overlap = float(np.prod(intersection) / max(1, np.prod(high - low)))
        if overlap < .65:
            return None
        aligned = self.recognizer.alignCrop(image, faces[0])
        return normalized_feature(self.recognizer.feature(aligned))

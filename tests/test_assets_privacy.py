from pathlib import Path
import tempfile
import unittest

import numpy as np

from facepass.diagnostics import DiagnosticFrames
from facepass.model_assets import load_verified_model


class AssetPrivacyTests(unittest.TestCase):
    def test_corrupt_model_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.task"
            path.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                load_verified_model(path)

    def test_explicit_diagnostic_directory_is_private_and_rate_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new"
            writer = DiagnosticFrames(path)
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            frame = np.zeros((100, 100, 3), np.uint8)
            writer.maybe_save(frame, 100)
            writer.maybe_save(frame, 100.1)
            writer.maybe_save(frame, 101)
            images = list(path.glob("*.png"))
            self.assertEqual(len(images), 2)
            for image in images:
                self.assertEqual(image.stat().st_mode & 0o077, 0)

    def test_existing_diagnostic_directory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                DiagnosticFrames(directory)


if __name__ == "__main__":
    unittest.main()

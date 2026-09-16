"""Opt-in biometric image retention. Never enabled automatically."""

import os
from pathlib import Path
import stat

import cv2


class DiagnosticFrames:
    def __init__(self, directory):
        self.directory = Path(directory)
        # Refuse existing directories, including symlinks and shared locations.
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        info = self.directory.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("diagnostic directory must have private POSIX permissions")
        self._last_saved = float("-inf")
        self._count = 0

    def maybe_save(self, frame, timestamp):
        if timestamp - self._last_saved < 1 or self._count >= 30:
            return
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("diagnostic encoding failed")
        path = self.directory / f"frame-{self._count:03d}.png"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            info = os.fstat(output.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("diagnostic image permissions are not private")
            output.write(encoded.tobytes())
        self._last_saved = timestamp
        self._count += 1

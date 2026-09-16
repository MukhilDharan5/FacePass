"""Direct V4L2 capture for an ordinary Fedora desktop user."""

from pathlib import Path
import re
import stat

import cv2

from .quality import validate_frame


class V4L2Camera:
    def __init__(self, device="/dev/video0", fps=12):
        if not re.fullmatch(r"/dev/video[0-9]+", device):
            raise ValueError("expected a local /dev/videoN device")
        if not stat.S_ISCHR(Path(device).stat().st_mode):
            raise ValueError("camera is not a character device")
        self._capture = cv2.VideoCapture()
        try:
            if not self._capture.open(device, cv2.CAP_V4L2):
                raise RuntimeError("camera open failed")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._capture.set(cv2.CAP_PROP_FPS, fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.close()
            raise

    def read(self):
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("camera read failed")
        validate_frame(frame)
        return frame

    def close(self):
        self._capture.release()

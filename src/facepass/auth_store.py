"""Administrator-owned enrollment templates; never retain camera images."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile

import numpy as np

from .auth_models import AUTH_MODELS, normalized_feature
from .evaluation import strict_json

STORE_PATH = Path("/var/lib/facepass/users")
MAX_TEMPLATE_BYTES = 80_000


def checked_uid(value):
    if type(value) is not int or not 1000 <= value < 2 ** 31:
        raise ValueError("face authentication requires a regular local user")
    return value


def validate_template(value, uid):
    fields = {"schema_version", "uid", "created_utc", "models", "templates"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid enrollment schema")
    if checked_uid(value["uid"]) != checked_uid(uid):
        raise ValueError("enrollment account mismatch")
    if value["models"] != {name: row[0] for name, row in AUTH_MODELS.items()}:
        raise ValueError("enrollment model mismatch")
    stamp = datetime.fromisoformat(value["created_utc"])
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("invalid enrollment timestamp")
    rows = value["templates"]
    if not isinstance(rows, list) or not 8 <= len(rows) <= 12:
        raise ValueError("enrollment requires 8..12 templates")
    for row in rows:
        if not isinstance(row, list) or len(row) != 128 or any(type(x) not in (int, float) for x in row):
            raise ValueError("invalid enrollment feature")
        feature = normalized_feature(row)
        if not np.allclose(feature, row, rtol=0, atol=1e-6):
            raise ValueError("enrollment features must be normalized")
    return value


class EnrollmentStore:
    def __init__(self, directory=STORE_PATH, *, owner_uid=0):
        self.path = Path(directory)
        self.owner_uid = owner_uid

    def _directory(self):
        info = self.path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != self.owner_uid
                or stat.S_IMODE(info.st_mode) != 0o700 or self.path.resolve() != self.path.absolute()):
            raise ValueError("enrollment directory must be owned, private and nonsymlinked")

    def read(self, uid):
        self._directory()
        path = self.path / f"{checked_uid(uid)}.json"
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != self.owner_uid
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError("insecure enrollment file")
            data = stream.read(MAX_TEMPLATE_BYTES + 1)
        if len(data) > MAX_TEMPLATE_BYTES:
            raise ValueError("oversized enrollment")
        return validate_template(strict_json(data), uid)

    def enrolled(self, uid):
        try:
            self.read(uid)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def write(self, uid, templates, *, replace=False):
        self._directory()
        if os.geteuid() != self.owner_uid:
            raise PermissionError("administrator authorization required")
        uid = checked_uid(uid)
        value = validate_template({"schema_version": 1, "uid": uid,
                                  "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "models": {name: row[0] for name, row in AUTH_MODELS.items()},
                                  "templates": [normalized_feature(row).tolist() for row in templates]}, uid)
        encoded = (json.dumps(value, allow_nan=False) + "\n").encode()
        if len(encoded) > MAX_TEMPLATE_BYTES:
            raise ValueError("oversized enrollment")
        target = self.path / f"{uid}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path, prefix=".enroll-", delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                if target.exists() or target.is_symlink():
                    self.read(uid)  # Refuse unsafe targets, even when replacing.
                os.replace(temporary, target)
            else:
                os.link(temporary, target, follow_symlinks=False)
                temporary.unlink()
            directory_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def remove(self, uid):
        self.read(uid)
        if os.geteuid() != self.owner_uid:
            raise PermissionError("administrator authorization required")
        (self.path / f"{checked_uid(uid)}.json").unlink()

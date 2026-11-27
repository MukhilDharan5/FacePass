"""Explicit setup download for pinned OpenCV identity assets; runtime is offline."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

from facepass.auth_models import AUTH_MODELS, verify_auth_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (digest, size) in AUTH_MODELS.items():
        target = args.directory / name
        if target.exists():
            verify_auth_model(target)
            print(f"Verified {name}", flush=True)
            continue
        folder = "face_detection_yunet" if "yunet" in name else "face_recognition_sface"
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{name}"
        with urllib.request.urlopen(url, timeout=60) as source:
            data = source.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("downloaded identity asset failed integrity verification")
        with tempfile.NamedTemporaryFile(dir=args.directory, delete=False) as output:
            output.write(data)
            temporary = Path(output.name)
        temporary.replace(target)
        verify_auth_model(target)
        print(f"Downloaded and verified {name}", flush=True)
        license_url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/{folder}/LICENSE"
        with urllib.request.urlopen(license_url, timeout=30) as source:
            (args.directory / f"{folder}.LICENSE").write_bytes(source.read(100_000))


if __name__ == "__main__":
    main()

"""Explicit one-time model download. The liveness runtime is always offline."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

# This script can run before installation, using Fedora's standard Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facepass.model_assets import MAX_MODEL_BYTES, MODEL_SHA256, MODEL_URL, load_verified_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/face_landmarker.task"))
    options = parser.parse_args()
    if options.output.exists():
        load_verified_model(options.output)
        print("Existing local model verified; no download needed.")
        return 0
    print("Downloading a model from Google; no camera data is accessed or transmitted.")
    with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
        data = response.read(MAX_MODEL_BYTES + 1)
    if len(data) > MAX_MODEL_BYTES or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("download integrity verification failed")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=options.output.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        # Atomic publication without overwriting an existing model.
        os.link(temporary, options.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Verified model saved to {options.output} (SHA-256 {MODEL_SHA256}).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Model setup failed: {type(error).__name__}. No unverified model was installed.", file=sys.stderr)
        raise SystemExit(2)

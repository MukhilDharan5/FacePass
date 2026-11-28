"""Prepare a self-contained, hash-locked offline installer from local wheels."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from packaging.utils import canonicalize_name, parse_wheel_filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dependency-wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    destination.mkdir(mode=0o700)  # Refuse an existing/possibly stale bundle.
    wheels = destination / "wheels"
    wheels.mkdir()
    locks = {}
    for line in (root / "requirements-fedora44.lock.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            locks[canonicalize_name(name)] = version
    locks["facepass-liveness"] = "0.14.0"
    candidates = list(args.dependency_wheels.glob("*.whl")) + [args.wheel]
    requirements = []
    for name, version in locks.items():
        selected = [path for path in candidates
                    if parse_wheel_filename(path.name)[0] == name and str(parse_wheel_filename(path.name)[1]) == version]
        if len(selected) != 1:
            raise ValueError(f"exactly one local wheel required for {name}=={version}")
        target = wheels / selected[0].name
        shutil.copyfile(selected[0], target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        requirements.append(f"{name}=={version} --hash=sha256:{digest}")
    (destination / "requirements-install.txt").write_text("\n".join(requirements) + "\n")
    files = ["tools/install_auth.py", "tools/auth_integration.py", "tools/setup_auth.py",
             "integration/facepass-auth.service", "integration/facepass-enroll.desktop",
             "native/pam_facepass.c", "native/Linux-PAM-Copyright", "docs/face-unlock.md",
             "models/face_landmarker.task", "models/face_detection_yunet_2023mar.onnx",
             "models/face_recognition_sface_2021dec.onnx", "models/face_detection_yunet.LICENSE",
             "models/face_recognition_sface.LICENSE"]
    files.extend(str(path.relative_to(root)) for path in (root / "native/include").rglob("*.h"))
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    manifest = {"version": "0.14.0", "sha256": {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob("*")) if path.is_file()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bundle": str(destination), "file_count": len(manifest["sha256"]),
                      "version": manifest["version"], "offline": True, "wheel_hashes_required": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

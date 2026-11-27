"""One-command laptop setup: administrator installation, then ordinary-user enrollment."""

import os
from pathlib import Path
import subprocess


def main():
    if os.geteuid() == 0:
        raise SystemExit("Open FacePass Setup as your desktop user.")
    bundle = Path(__file__).resolve().parents[1]
    python = Path("/opt/facepass-auth/current/venv/bin/python")
    if not python.exists():
        print("FacePass 0.14.0 — Fedora KDE face-unlock setup", flush=True)
        print("This installs the local camera service, PAM bridge and visible lock-screen button.", flush=True)
        print("Enter your laptop administrator password at the sudo prompt. No password is sent to Codex.", flush=True)
        result = subprocess.run(["/usr/bin/sudo", "--", "/usr/bin/python3", "-I", str(bundle / "tools/install_auth.py"),
                                 "--install", "--bundle", str(bundle), "--login"])
        if result.returncode:
            print("Installation did not finish. Password login remains available. Review the error above and rerun FacePass Setup.", flush=True)
            return result.returncode
    print("Opening enrollment. Follow the administrator dialog and both camera challenges.", flush=True)
    return subprocess.run([str(python), "-I", "-m", "facepass.enrollment_ui"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())

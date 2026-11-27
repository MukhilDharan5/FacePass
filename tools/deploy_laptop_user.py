"""Upgrade the existing per-user install and expose the persistent system setup."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        parser.error("run as the ordinary desktop user")
    home = Path.home()
    root = home / ".local/share/facepass"
    bundle = root / "system-setup/0.14.0"
    if bundle.exists():
        raise FileExistsError("Persistent setup already exists; refusing to replace it.")
    bundle.parent.mkdir(mode=0o700, exist_ok=True)
    shutil.copytree(args.bundle, bundle)
    bundle.chmod(0o700)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
            raise ValueError("persistent bundle integrity failure")
    wheel = bundle / "wheels/facepass_liveness-0.14.0-py3-none-any.whl"
    shutil.copyfile(wheel, root / wheel.name)
    python = root / "venv/bin/python"
    subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", str(root / wheel.name)],
                   check=True, env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True)
    commands = home / ".local/bin"
    setup = commands / "facepass-setup"
    setup.write_text('#!/bin/sh\nexec /usr/bin/python3 -I ' + shlex.quote(str(bundle / "tools/setup_auth.py")) + ' "$@"\n')
    setup.chmod(0o700)
    auth = commands / "facepass-auth"
    auth.write_text('#!/bin/sh\nexec ' + shlex.quote(str(python)) + ' -I -m facepass.auth_cli "$@"\n')
    auth.chmod(0o700)
    enroll = commands / "facepass-enroll"
    enroll.write_text('#!/bin/sh\nexec ' + shlex.quote(str(setup)) + ' "$@"\n')
    enroll.chmod(0o700)
    desktop = home / ".local/share/applications/facepass-enroll.desktop"
    desktop.write_text('[Desktop Entry]\nType=Application\nName=FacePass Setup\nComment=Install face unlock and enroll your face\nExec=/usr/bin/konsole --separate --hold -e '
                       + str(setup) + '\nIcon=camera-web\nTerminal=false\nCategories=Settings;Security;\nStartupNotify=true\n')
    desktop.chmod(0o600)
    receipt_path = root / "install.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(version="0.14.0", previous_version=receipt["version"], updated_at=datetime.now(timezone.utc).isoformat(),
                   system_setup_bundle=str(bundle), system_setup_ready=True,
                   authentication_enabled=False, system_authentication_activation_pending=True,
                   patch="Experimental KDE/PAM face unlock implemented; offline administrator installation and physical enrollment required")
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as temporary:
        json.dump(receipt, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary.name, receipt_path)
    subprocess.run(["/usr/bin/kbuildsycoca6", "--noincremental"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(json.dumps({"installed_user_version": "0.14.0", "system_bundle": str(bundle), "setup_command": str(setup),
                      "desktop_launcher": str(desktop), "authentication_activation_pending": True}))
    if args.launch:
        subprocess.Popen(["/usr/bin/konsole", "--separate", "--hold", "-e", str(setup)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("Opened FacePass Setup in a new Konsole window for the laptop administrator prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

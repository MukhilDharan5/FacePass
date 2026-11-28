"""Offline administrator installer. Inspect with --check; undo with --uninstall."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

# Isolated Python deliberately excludes the script directory from sys.path.
_spec = importlib.util.spec_from_file_location("facepass_install_edits", Path(__file__).with_name("auth_integration.py"))
_edits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_edits)
patch_main, patch_messages, patch_pam, remove_block = _edits.patch_main, _edits.patch_messages, _edits.patch_pam, _edits.remove_block

BASE = Path("/opt/facepass-auth")
STATE = Path("/var/lib/facepass/install.json")
SHELL = Path("/usr/share/plasma/shells/org.kde.plasma.desktop/contents/lockscreen")
UNIT = Path("/etc/systemd/system/facepass-auth.service")
MODULE = Path("/usr/lib64/security/pam_facepass.so")
VERSION = "0.14.0"


def run(command):
    subprocess.run(command, check=True, env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                                           "LANG": "C.UTF-8", "HOME": "/root"})


def atomic(path, data, mode=0o644):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        os.fchmod(stream.fileno(), mode)
        stream.write(data if isinstance(data, bytes) else data.encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan(login=False):
    edits = {Path("/etc/pam.d/kde"): patch_pam(Path("/etc/pam.d/kde").read_text()),
             SHELL / "MainBlock.qml": patch_main((SHELL / "MainBlock.qml").read_text()),
             SHELL / "LockScreenUi.qml": patch_messages((SHELL / "LockScreenUi.qml").read_text())}
    originals = {str(path): {"existed": path.exists(), "text": path.read_text()}
                 for path in edits}
    if login:
        for service in ("plasmalogin", "sddm"):
            target = Path("/etc/pam.d") / service
            vendor = Path("/usr/lib/pam.d") / service
            source = target if target.exists() else vendor
            if source.exists():
                edits[target] = patch_pam(source.read_text())
                originals[str(target)] = {"existed": target.exists(), "text": source.read_text()}
    if any(path.is_symlink() for path in edits):
        raise ValueError("PAM/theme targets must not be symlinks")
    return edits, originals


def verify_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ValueError("unexpected installer bundle")
    for name, digest in manifest["sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid bundle path")
        path = bundle / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.resolve() != path.absolute():
            raise ValueError("bundle contains symlink/nonregular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"bundle integrity failure: {name}")
    return manifest


def install(bundle, login):
    if os.geteuid() != 0:
        raise PermissionError("Run this installer through the laptop administrator prompt.")
    if STATE.exists():
        print("FacePass is already installed. Use its setup app to enroll, or uninstall before replacing the system component.")
        return
    edits, originals = plan(login)
    manifest = verify_bundle(bundle)
    for program in ("gcc", "systemctl", "restorecon"):
        if not shutil.which(program):
            raise RuntimeError(f"required system command is missing: {program}")
    if MODULE.exists() or UNIT.exists() or BASE.exists():
        raise FileExistsError("existing authentication files require review before installation")
    os.umask(0o022)
    state = {"version": VERSION, "originals": originals, "patched": {}, "created": []}
    state_directory = STATE.parent
    if state_directory.exists() and (state_directory.is_symlink() or state_directory.stat().st_uid != 0):
        raise ValueError("insecure authentication state directory")
    state_directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(state_directory, 0o700)
    users = state_directory / "users"
    users.mkdir(mode=0o700, exist_ok=True)
    if users.is_symlink() or users.stat().st_uid != 0:
        raise ValueError("insecure enrollment directory")
    os.chmod(users, 0o700)
    BASE.mkdir(mode=0o755)
    release = BASE / "releases" / VERSION
    release.mkdir(parents=True)
    payload = release / "payload"
    payload.mkdir(mode=0o700)
    try:
        # Runtime and build tools consume a root-owned snapshot after validation.
        for name in manifest["sha256"]:
            source = bundle / name
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
                raise ValueError("bundle changed while copying")
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic(target, data)
        run(["/usr/bin/python3", "-I", "-m", "venv", str(release / "venv")])
        python = str(release / "venv/bin/python")
        run([python, "-I", "-m", "pip", "install", "--no-index", "--find-links", str(payload / "wheels"),
             "--require-hashes", "-r", str(payload / "requirements-install.txt")])
        run([python, "-I", "-m", "pip", "check"])
        shutil.copytree(payload / "models", release / "models")
        run(["/usr/bin/gcc", "-std=c11", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror",
             "-D_FORTIFY_SOURCE=3", "-fstack-protector-strong", "-Wl,-z,relro,-z,now,-z,noexecstack",
             "-I", str(payload / "native/include"), str(payload / "native/pam_facepass.c"),
             "-l:libpam.so.0", "-o", str(release / "pam_facepass.so")])
        shutil.copy2(payload / "tools/install_auth.py", BASE / "install_auth.py")
        shutil.copy2(payload / "tools/auth_integration.py", BASE / "auth_integration.py")
        (BASE / "current").symlink_to(release)
        run([python, "-I", "-c", "from facepass.auth_models import IdentityModel; IdentityModel(); print('Identity models verified.')"])
        atomic(MODULE, (release / "pam_facepass.so").read_bytes(), 0o755)
        state["created"].append(str(MODULE))
        atomic(UNIT, (payload / "integration/facepass-auth.service").read_bytes())
        state["created"].append(str(UNIT))
        for name, module in (("facepass-auth", "facepass.auth_cli"), ("facepass-enroll", "facepass.enrollment_ui")):
            path = Path("/usr/local/bin") / name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"command already exists: {path}")
            atomic(path, '#!/bin/sh\nexec /opt/facepass-auth/current/venv/bin/python -I -m ' + module + ' "$@"\n', 0o755)
            state["created"].append(str(path))
        desktop = Path("/usr/share/applications/facepass-enroll.desktop")
        if desktop.exists():
            raise FileExistsError("FacePass Setup desktop entry already exists")
        atomic(desktop, (payload / "integration/facepass-enroll.desktop").read_bytes())
        state["created"].append(str(desktop))
        run(["/usr/sbin/restorecon", "-RF", str(BASE), str(state_directory), str(MODULE), str(UNIT)])
        run(["/usr/bin/systemctl", "daemon-reload"])
        run(["/usr/bin/systemctl", "enable", "--now", "facepass-auth.service"])
        run(["/usr/bin/systemctl", "is-active", "--quiet", "facepass-auth.service"])
        # Verify a real root-peer IPC round trip before changing any PAM file.
        probe = """from facepass.auth_cli import auth_request
import time
for attempt in range(50):
    try:
        print(list(auth_request('STATUS',1000)))
        break
    except OSError:
        if attempt == 49: raise
        time.sleep(.1)
"""
        run([python, "-I", "-c", probe])
        for path, text in edits.items():
            atomic(path, text)
            state["patched"][str(path)] = hashlib.sha256(text.encode()).hexdigest()
        run(["/usr/sbin/restorecon", "-RF", "/etc/pam.d"])
        atomic(STATE, json.dumps(state, indent=2) + "\n", 0o600)
        shutil.rmtree(payload)
        print("FacePass installed. Open FacePass Setup, enroll and complete both camera challenges. Password login remains available.")
    except BaseException:
        # Complete rollback of edits made during this attempt, even before a receipt exists.
        rollback(state, remove_install=True)
        raise


def rollback(state, remove_install=True):
    subprocess.run(["/usr/bin/systemctl", "disable", "--now", "facepass-auth.service"], check=False)
    for name, original in state["originals"].items():
        path = Path(name)
        if name not in state["patched"] or not path.exists():
            continue
        current = path.read_text()
        if hashlib.sha256(current.encode()).hexdigest() == state["patched"][name]:
            if original["existed"]:
                atomic(path, original["text"])
            else:
                path.unlink()
        else:
            # Preserve subsequent administrator/distribution edits.
            atomic(path, remove_block(current, "qml" if path.suffix == ".qml" else "pam"))
    for name in state["created"]:
        path = Path(name)
        if path.exists():
            path.unlink()
    if STATE.exists():
        STATE.unlink()
    if remove_install and BASE.exists():
        shutil.rmtree(BASE)
    subprocess.run(["/usr/bin/systemctl", "daemon-reload"], check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--install", action="store_true")
    action.add_argument("--uninstall", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--login", action="store_true", help="Also integrate the installed Plasma Login/SDDM manager")
    args = parser.parse_args(argv)
    if args.check:
        edits, _ = plan(args.login)
        if args.bundle:
            verify_bundle(args.bundle.resolve())
        print(json.dumps({"version": VERSION, "edit_paths": [str(path) for path in edits],
                          "requires_administrator": True, "enrollment_required": True,
                          "password_fallback": True, "runtime_network": False}, indent=2))
        return 0
    if os.geteuid() != 0:
        parser.error("system integration requires laptop administrator authorization")
    if args.uninstall:
        if not STATE.exists():
            parser.error("no FacePass system installation receipt")
        rollback(json.loads(STATE.read_text()))
        print("FacePass system integration removed. Private enrollment templates retained; password login restored.")
    elif args.bundle:
        install(args.bundle.resolve(), args.login)
    else:
        parser.error("--bundle is required for installation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Fedora KDE Plasma setup

FacePass's terminal capture, evaluation tools and local research daemon work
independently of the desktop. KDE Plasma uses the same ordinary-user V4L2 adapter
and supervised worker as GNOME. The optional preview uses Tk and XWayland in a
Plasma Wayland session. Fedora's [Plasma 6 change](https://fedoraproject.org/wiki/Changes/KDE_Plasma_6)
retains XWayland compatibility for X11 applications; changing the desktop to an
X11 session is unnecessary.

## Install and run

This laptop now has an independent user installation with an application-menu
launcher and commands available in Konsole; see [laptop installation](laptop-installation.md).
The following steps describe setting up a development checkout on another machine.

These commands target the regular Fedora KDE Plasma Desktop edition with DNF and
Fedora's system Python. Open **Konsole**, clone the project, and enter its directory:

```bash
git clone https://github.com/MukhilDharan5/FacePass.git
cd FacePass
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fedora44.lock.txt -e .
.venv/bin/python tools/fetch_model.py
.venv/bin/python -m facepass --doctor
```

An existing checkout only needs its environment and model setup completed.
The lock file records the Fedora 44 reference environment; the project's
`pyproject.toml` defines its Python requirements. Model and dependency downloads
occur only during these explicit setup steps.

Choose a readable/writable video capture device listed by doctor:

```bash
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Capture returns status **2**, including a clean observation: this is research
diagnostics with authentication disabled. Doctor and other successful tooling
commands return 0. Terminal capture needs neither Tk nor a display connection.

## Optional preview under Plasma Wayland

Install Fedora's [Tkinter package](https://packages.fedoraproject.org/pkgs/python3.14/python3-tkinter/)
and [XWayland package](https://packages.fedoraproject.org/pkgs/xorg-x11-server-Xwayland/xorg-x11-server-Xwayland/)
if they are missing:

```bash
sudo dnf install python3-tkinter xorg-x11-server-Xwayland
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --preview
```

Use Konsole inside your logged-in Plasma session. Tk uses the inherited `DISPLAY`
to reach XWayland; `WAYLAND_DISPLAY` alone cannot supply Tk's X11 connection.
The preview does not use OpenCV's Qt windows and does not change
`QT_QPA_PLATFORM`, `QT_QPA_PLATFORMTHEME`, `DISPLAY` or desktop settings. Close the
window or press Escape to cancel. Preview images stay in memory unless you
explicitly request diagnostic frame retention.

Doctor adds `desktop_environment` (`kde-plasma` for KDE/Plasma hints), display
configuration flags and a `preview` object to its existing output. These are
environment diagnostics, not proof that a physical desktop is accessible:

| `preview.status` | Meaning and action |
| --- | --- |
| `ready_to_try` | Tk's module and `DISPLAY` are present. Try `--preview`; a stale or unauthorized display can still fail. |
| `tkinter_missing` | Install Tk for the interpreter used by `.venv`. `python3-tkinter` serves Fedora's system Python; a custom Python needs its matching Tk installation. |
| `xwayland_display_missing` | A Wayland session is indicated but `DISPLAY` is missing. Check XWayland is installed and launch from Konsole in the active Plasma session. |
| `display_missing` | No X11 display is configured. Run in a desktop terminal for preview, or omit `--preview` for terminal capture. |

`preview.backend` is `xwayland`, `x11` or `unavailable`. An existing X11 session
also uses the same Tk preview; Fedora Plasma Wayland is the setup target.
`tk_preview_available` retains its earlier meaning: the Tkinter module can be
found, without testing window initialization or its native dependencies.

For a missing camera, select another capture node if the webcam exposes multiple
nodes; metadata nodes cannot deliver frames. Stop other applications using that
camera and rerun doctor to inspect existing access permissions. The engine reads
`/dev/videoN` directly and does not use the KDE camera portal.

## Daemon and evaluation

The [daemon and client](daemon-protocol.md) share the normal user session's
`XDG_RUNTIME_DIR`, including under Plasma. Run the foreground daemon in one
Konsole tab and the client in a second tab under the same user/environment:

```bash
.venv/bin/python -m facepass.daemon
# In the second tab:
.venv/bin/python -m facepass.client status
.venv/bin/python -m facepass.client observe --request-id trial001 --duration 5
```

The [evaluation workflow](evaluation-tooling.md) and `--profile` commands work the
same way from Konsole. No desktop autostart or system service is installed.

## Interpreting capture output

`faces=1 landmarks=478` means the camera and landmark model found a face.
`track=...:STABLE` means usable observations have remained continuous; it does
not establish identity or liveness. Signal confidence 0.50 is neutral, while
`available=False` means the signal cannot currently supply evidence. This research
version keeps overall confidence at zero and never grants authentication.

`INSUFFICIENT_QUALITY (observation_timeout)` means the observation window ended
with the latest frame failing a quality gate. Read the final `quality` line for
the cause. With `blur`, try brighter, even lighting, a clean camera lens and a
steady face at a comfortable distance. The terminal reports the measured
`blur_variance`, `mean_luminance` and face width when available. Camera-specific
quality thresholds still require physical evaluation.

Version 0.13.0 defaults to the `tolerant` quality profile. Moderate blur, dim detail
and smaller resolved faces can remain usable with warnings instead of clearing
the track. Final output includes raw and evaluated sharpness, contrast and
low-light landmark enhancement state. `--quality-profile strict` restores the
original gates. See [capture quality profiles](capture-quality.md) for the limits.

Version 0.12.1 fixes a misleading cleanup label: implemented detectors whose
history was cleared after quality loss report `quality_not_acceptable`, rather
than claiming that the detectors are unimplemented. Closed clean sessions retain
only explicitly historical scalar diagnostics. Neither path can grant access.

For a longer observation and scalar timing diagnostics:

```bash
facepass --device /dev/video0 --duration 8 --preview --profile
```

## Verification boundary

All 248 unit/integration tests and all 12 offline verification checks passed on
Fedora 44/Python 3.14.8 with KDE session settings for version 0.12.0. The
[verification reference](kde-verification-reference.jsonl) records the run.
Automated tests cover KDE/Plasma environment hints, Wayland/XWayland and X11
preview prerequisites, GNOME compatibility, absent display/Tk, preview failures
before capture, and terminal operation without any GUI requirement. Real CLI
checks also covered those desktop hints, missing-display preview failure and the
supervised missing-camera path under displayless KDE settings.

The subsequent 0.12.1 diagnostic patch passed all 250 regression tests and the
standalone installed-package checks. Its quality-loss timeout regression also
verifies that history is cleared and implemented detectors are labeled unavailable.

The current environment advertises KDE/Wayland. A real Tk connection, hidden
generated-PNG preview, event processing and cleanup passed with Tk reporting the
`x11` windowing system, consistent with XWayland. The sandbox initially blocked
the display connection; the same check succeeded outside its socket restriction.
The [display reference](kde-display-reference.json) contains only scalar results.
To repeat this optional check in a desktop terminal without opening a camera:

```bash
.venv/bin/python tools/smoke_preview.py
```

No physical webcam is available here. An end-to-end camera preview, visible-window
layout and physical-camera check remain necessary on the target machine. The
display smoke check does not validate camera capture or attack resistance.

The research harness keeps authentication disabled. Version 0.14.0 adds an
independent experimental KDE lock-screen/PAM authenticator, enrollment setup,
and optional Plasma Login/SDDM integration. See [face-unlock.md](face-unlock.md).

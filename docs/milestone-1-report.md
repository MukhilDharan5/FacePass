# Milestone 1 implementation report

Implemented on 2026-09-16 for Fedora GNOME, in the FacePass workspace.

## What changed and why

The previously empty project now has a standalone Python package, an ordinary-user
V4L2 camera adapter, verified local MediaPipe landmarks, experimental quality gates
and a finite diagnostic harness. The public session API and explicit unavailable
signal slots let later detectors be introduced separately. Success is disabled so
good landmarks, capture quality or clean process exit cannot imply liveness.

Camera/model work runs in a supervised child to bound blocking native calls.
Cancellation, missing frames, malformed IPC, model errors, multiple detected faces
and cleanup/crash failures produce non-success. Frames stay transient unless the
developer explicitly enables warned, private, bounded diagnostic retention.

## Assumptions and failure cases

Target: Fedora 44 Workstation, GNOME Wayland, CPython 3.14.7 on x86_64, an existing
readable/writable V4L2 RGB camera and a nearby face. Tested dependencies are recorded
in requirements-fedora44.lock.txt. Hardware capture/preview is still unverified.

Quality thresholds are experimental. Occlusion and calibrated mesh confidence are
unavailable; orientation is a rough nose/eye proxy. Small/occluded/profile faces,
glasses, poor lighting and sharpened/noisy captures can defeat detection or quality
gates. A missed second face is not known to the engine. Tracking continuity, blink,
gaze, head motion, lighting PAD, texture PAD, temporal fusion, adaptive positive
decisions and challenge checking remain future work. Any photo/video producing
valid landmarks still cannot yield LIVE in this version.

## Verification performed

| Check | Outcome |
| --- | --- |
| 30 synthetic/security-boundary unit tests | Passed |
| Python source compilation | Passed |
| Real pinned model initialization, blank-frame inference and close | Passed; zero faces, INSUFFICIENT_QUALITY |
| Missing-camera terminal harness | UNCERTAIN, camera_open_failed, confidence 0, exit 2 |
| Installed dependency consistency | pip check passed |
| Existing model setup helper | SHA-256 verified locally, no download required |
| Physical webcam, real people and physical attacks | Not tested: no /dev/video* visible |
| GNOME preview | Not tested: tkinter unavailable |

No biometric frames or identity data were collected for automated verification.
Temporary synthetic diagnostic images used in tests were deleted with their test
directories. No authentication configuration or system packages were changed.

## How to test on the laptop

From the project directory:

```bash
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --preview
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_model.py
```

Choose the camera node reported by doctor; omit preview without Tk. Run the manual
matrix in evaluation.md, paying particular attention to multiple faces, quality
reasons, loss/re-entry and cancellation. Every capture attempt must remain non-success.

Development stops here as the user brief explicitly requires. Milestone 2 is stable
multi-frame face tracking and evidence reset on loss/discontinuity; it needs new
direction before implementation begins.

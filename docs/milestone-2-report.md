# Milestone 2: stable multi-frame continuity tracking

Implemented for Fedora GNOME in FacePass on 2026-09-16. This milestone adds
conservative short-term continuity to the local camera/landmark foundation;
**liveness remains unavailable and every capture run remains non-success.**

## What was implemented

`FaceTracker` associates consecutive single-face, usable captures using center
displacement, scale change, relatively rigid landmark shape and a small aligned
appearance feature. Shape comparison removes translation, isotropic scale and
roll. An initial anchor and the preceding observation help detect drift or abrupt
replacement. This does not infer depth or perform face recognition.

SEARCHING → ACQUIRING → STABLE requires at least three consecutive usable frames
spanning at least 150 ms by default. Extremely still faces can stabilize. Eye/iris
and mouth changes are not used as continuity shape anchors, so blinking or subtle
gaze does not itself break a track. All original landmark samples are preserved
without temporal smoothing for later detectors.

History is owned, immutable geometry, bounded to 64 samples and two seconds by
default, with no retained previous full frame. The tracker retains one initial
geometry anchor and small grayscale appearance features in private memory. Neither
history nor these features appears in public diagnostics.

| Condition | Implemented behavior |
| --- | --- |
| Consecutive compatible usable observations | Acquire, then STABLE; liveness stays UNCERTAIN |
| No face or unacceptable quality | Clear history/anchor/appearance/evidence; LOST or SEARCHING |
| Face returns after loss | New attempt-local track number, restart ACQUIRING |
| Gap over 350 ms | Clear history, including when accessed without a new frame |
| Short gap followed by fresh quality capture | Start a fresh track, never interpolate old evidence |
| Session frame gap over one second | Existing terminal capture failure |
| Strong position/scale/shape/appearance discontinuity | Clear retained data, terminate UNCERTAIN with mismatch reason |
| Camera resolution change | Clear retained data, terminate UNCERTAIN |
| Multiple detected faces | Existing terminal ambiguity refusal, clear history/evidence |
| Model/camera/frame error, malformed data, cancellation or finish | Clear retained data and close backend |

The session clears all existing signal slots when history becomes invalid. Future
stateful liveness detectors must also reset their own caches at these boundaries.
Worker JSON validation now checks tracking states/counters/finite measurements and
rejects unknown fields, including attempted vector data. Preview work rechecks
freshness before emitting diagnostics. The terminal/preview exposes track state,
track number, history/reset counts and reasons; JSON includes scalar continuity
measurements. No tracking state can override the prohibition on LIVE.

## Assumptions and limits

Target remains Fedora 44 Workstation GNOME Wayland, CPython 3.14.7 x86_64, trusted
local model/camera/OS and an ordinary-user V4L2 RGB camera. No system packages or
PAM/GDM/GNOME login configuration were changed. Detection continues over the full
frame on every processed capture to detect new nearby faces.

Tracking thresholds are experimental continuity gates. Similar-looking replacement
faces, prints, masks, gradual substitutions and coherent video may remain compatible.
STABLE means compatible observations, not proof of identity, a live 3D person or
attack resistance. A session-local track number cannot bind an enrolled identity.
Physical face-switch performance remains untested.

Glasses, directional lighting changes, shadows, reflections, rapid head/camera
motion and noisy landmarks may produce conservative UNCERTAIN results. Uniform
brightness/positive contrast changes are normalized in appearance comparison;
unusable texture makes appearance unavailable and caps geometric continuity score
at 0.5 instead of arbitrarily rejecting the face. The continuity score is not a
calibrated probability and must not become an independent PAD evidence producer.

Consumers accessing `session.history` must discard references after use/reset;
clearing the engine's ring cannot erase references a caller chose to retain.
Detection misses remain possible, including an occluded or tiny second face.
Occlusion, calibrated landmark confidence and measured depth remain unavailable.

## Verification performed

| Check | Result |
| --- | --- |
| 56 synthetic/security-boundary unit tests | Passed |
| Still face without blink/motion | STABLE with unchanged track number, never LIVE |
| Gradual translation/scale/roll, landmark jitter, eye/iris changes | Continuity maintained in synthetic cases; raw geometry retained unchanged |
| Strong same-geometry appearance substitution | Terminal UNCERTAIN, appearance_discontinuity, empty history |
| Strong geometry substitution | Rejected and history cleared in synthetic cases |
| Loss/quality failure/gap/staleness | History/evidence cleared; reacquisition starts a new track |
| Bounded immutable samples and malformed input after stability | Verified; malformed input rejects/clears the tracker |
| Offline synthetic tracking demo | Still face → stable → loss → new track → appearance rejection; all non-success |
| Real model smoke check on synthetic blank pixels | Loads, infers zero faces, closes successfully |
| Missing-camera supervised harness | UNCERTAIN, camera_open_failed, confidence 0, empty tracking history, exit 2 |
| Physical webcam and real user/attack sequences | Not tested: no /dev/video* visible here |
| GNOME Tk preview | Not tested: tkinter unavailable here |

These are engineering checks, not measured PAD false-acceptance/rejection results.
No biometric images or identity data were collected for verification.

## How to test

From the project directory:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_tracking.py
.venv/bin/python tools/smoke_model.py
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --preview
```

Choose the camera capture node listed by doctor and omit preview when Tk is
unavailable. The synthetic demo opens no webcam, prints scalar signal/tracking JSON
and asserts reset behavior; exit 0 means only that its engineering checks passed.
Normal camera runs still exit 2.

On hardware, look naturally at the camera, remain still, then try small ordinary
movements. Inspect acquisition/stability/reset reasons. Leave/re-enter the frame:
history must clear and a new track must reacquire. Try glasses, side lighting and
the full evaluation.md matrix. Two visible credible faces must terminate the
attempt when detected. Ctrl+C or closing preview must stop camera work promptly.
Every scenario must remain non-success at this milestone.

Development pauses here as requested by the original brief. Milestone 3 is subtle
head/landmark motion evidence and planar/background motion comparison; it has not
been started.

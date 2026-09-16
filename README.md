# FacePass

Local, passive RGB-camera liveness and presentation-attack-detection research for
Fedora GNOME. The engine is independent of desktop authentication and processes
camera data locally.

## Current progress

**Version 0.3.0 · milestones 1–3 implemented · 79 automated tests passing.**

Milestones 1–3 implement local webcam capture, face landmarks, short-term face
continuity and experimental head-motion/geometry evidence. **Validated liveness
and confidence fusion are not implemented. This version never returns LIVE or a
successful authentication exit status.** Nothing in this project installs PAM modules, a service, a GDM hook,
a GNOME extension, enrollment data, or login configuration.

| Milestone | Delivered | Validation still needed |
| --- | --- | --- |
| 1 — Camera and landmarks | V4L2 capture, local face mesh, quality gates, supervised diagnostic harness | Physical webcam capture and GNOME preview |
| 2 — Stable multi-frame tracking | One-face continuity, immutable bounded history, discontinuity refusal and reset handling | Physical face-switch, loss and reacquisition scenarios |
| 3 — Head micro-motion | Approximate pose, small landmark movement, planar fit, background compensation and two weak evidence producers | Physical motion sensitivity and print/display/replay evaluation |

The implementation has been checked on Fedora 44 Workstation, GNOME Wayland,
x86_64 and Python 3.14.7. No camera is visible in the development execution
environment and Tk is unavailable, so hardware and preview tests are outstanding.
Passing synthetic/unit tests does not establish attack resistance.

The supplied brief requires a pause after each major milestone. Development is
paused after milestone 3; milestone 4 is next and has not started.

## Future plan

| Milestone | Planned work | Required behavior |
| --- | --- | --- |
| 4 — Blink sequences | Temporal eyelid closure/opening state machine, bilateral checks, duration and noise handling | Distinguish blinks, sustained closure and tracking failure; no mandatory blink |
| 5 — Gaze movement | Iris position relative to eyes and head over time | Reduce reliability for glasses, poor visibility and insufficient resolution |
| 6 — Illumination, experimental | Multi-region facial shading/color and ambient scene response across time | Account for exposure/white balance; avoid a simple face/background brightness comparison |
| 7 — Replay/print evidence, experimental | Texture, frequency, banding, moire, halftoning, glare and resampling diagnostics | Treat missing artifacts and camera-specific confounders conservatively |
| 8 — Temporal consistency and fusion | Combine quality-gated signal histories with explicit dependence groups | Retain individual scores; never grant LIVE from one heuristic |
| 9 — Adaptive observation | Validated confidence policy, early completion, bounded extensions and optional challenge support | Hard timeout, passive default and UNCERTAIN when evidence is inadequate |
| 10 — Attack evaluation | Real users, stillness, glasses, lighting, prints, displays, replays, moving media and failures | Held-out users/rooms/media; separate false acceptance, false rejection and nonresponse rates |
| 11 — CPU and latency | Profile laptop workloads, tune detector cadence and cancellation | Low idle cost and bounded active work without weakening validated policy |
| 12 — Daemon interface | Versioned local request/result/cancellation contract after extensive evaluation | Keep identity recognition separate and preserve fail-closed handling |

Each milestone includes tests, an implementation report, assumptions and known
failure cases, followed by a pause for direction. Physical capture checks should
begin as soon as a webcam is available, and attack evaluation must inform future
calibration. LIVE stays disabled until diverse evidence, quality and fusion policy
are validated; implementing more detectors alone does not enable success.
PAM/GDM/GNOME integration remains a separate future project, outside this scope.

Milestone reports: [1](docs/milestone-1-report.md),
[2](docs/milestone-2-report.md), [3](docs/milestone-3-report.md).

## Implemented capabilities

- An ordinary-user `/dev/videoN` V4L2 capture adapter, requesting 640 × 480 at
  12 processed frames per second by default. Device settings are best effort.
- Local MediaPipe Face Landmarker inference with 478 landmarks per detected face,
  including eye/iris geometry for later work. Up to four faces are detected afresh
  on each frame. Two or more detected faces terminate the attempt as ambiguous.
- Experimental gates for face size, clipping, blur, exposure and rough orientation.
  Zero faces and poor quality produce INSUFFICIENT_QUALITY; camera/model failures
  produce UNCERTAIN. Quality never supplies liveness evidence.
- A bounded session API with transient geometry and scalar diagnostics. Head
  motion and facial geometry have weak experimental evidence producers; the
  other five signal slots remain unavailable with zero reliability.
- Consecutive quality-gated face tracking, with position/scale limits, landmark
  shape alignment and a small in-memory appearance comparison. A still face can
  become STABLE without blinking or moving. STABLE does not establish identity
  or liveness, and the tracking score is not a PAD signal.
- Immutable, bounded landmark history (at most 64 samples and two seconds by
  default). Loss, poor quality, stale capture and failures clear history/evidence;
  reacquisition gets a fresh session-local track number. A strong visible
  discontinuity terminates the attempt as UNCERTAIN.
- Approximate yaw/pitch/roll, short-window landmark displacement, held-out planar
  transform error and sparse background-relative motion. Evidence requires a
  stable quality track and at least six frames spanning 450 ms. Stillness, noisy
  motion, shared scene movement and an adequate planar explanation stay neutral;
  no deliberate head turn or other challenge is required.
- Two replaceable motion evidence functions, capped at confidence 0.68 and
  reliability 0.4 or below. These are uncalibrated engineering scores, share
  measurements and cannot be treated as independent security factors.
- A finite worker supervised by the terminal harness, with a startup limit,
  capture/model stall detection, JSON validation, crash detection and cancellation.
- Optional annotated preview and explicit, disabled-by-default diagnostic image
  retention. No runtime downloads, network calls or identity recognition.

## Use on Fedora GNOME

Run as your normal user from GNOME Terminal in the project directory. For a fresh
clone, first follow the environment setup below; `.venv` and the model are not
included in the repository. Start with:

```bash
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5
```

The doctor command does not open the webcam. Select a readable/writable capture
node from its camera list. Some webcams expose a separate metadata node which
cannot deliver video; try the other capture node if opening/reading fails. A
missing or unavailable camera safely ends the attempt.

Every capture run ends with the final decision and individual signal confidence,
reliability and availability. `--json` emits structured scalar updates and the
final result, including tracking state, current session-local track number, history
count, reset count/reason, approximate pose and scalar motion diagnostics.
Confidence 0.5 denotes neutral evidence; none of these heuristic signal scores
is a measured liveness probability. Overall liveness confidence stays zero.
On a clean finish, motion values are marked historical, with signal availability
false and reliability zero. Images, geometry and motion caches are released.
Failure, cancellation, quality loss and staleness clear even those motion values.

The harness exits with code **2** for every capture result at this milestone,
including a clean quality capture. `--doctor`, the unit tests, and the model setup
helper may exit 0 for their respective tooling success; they are not authentication
entry points. Ctrl+C cancels capture; no pipeline runs after the harness exits.

Optional in-memory landmarks preview:

```bash
.venv/bin/python -m facepass --device /dev/video0 --preview
```

Preview uses Tk rather than OpenCV's Qt window to avoid coupling the camera engine
to Qt platform plugins. Fedora's `python3-tkinter` package and a desktop display are
required for the system Python. Tk uses XWayland in a GNOME Wayland session. If Tk
is missing, run terminal mode; the preview does not silently fall back to recording.
Close the window or press Escape to cancel. Terminal mode needs no display server.

This prototype reads V4L2 devices directly. It is not a Flatpak or a PipeWire camera
portal client, and does not run at GDM or the lock screen. Keep SELinux enforcing
and use existing desktop camera permissions. Do not run it as root, change device
permissions, add authentication hooks or disable SELinux to make a test work.

## Recreate the local development environment

Only these explicit setup steps need internet access, to obtain dependencies and
the model. Camera processing is offline. Use an isolated environment; Fedora's
[Python guide](https://developer.fedoraproject.org/tech/languages/python/python-installation.html)
describes this workflow.

```bash
git clone https://github.com/MukhilDharan5/FacePass.git
cd FacePass
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fedora44.lock.txt -e .
.venv/bin/python tools/fetch_model.py
```

The version file records dependencies tested on Fedora 44 x86_64, Python 3.14.7.
Other Fedora/Python versions are unverified; the project metadata provides version
ranges for fresh resolution. Do not install multiple OpenCV wheel variants in the
same environment: MediaPipe depends on `opencv-contrib-python`, which this project
also uses. No system Python packages were changed during implementation.

The setup helper downloads Google's version-1 Face Landmarker task bundle only
when explicitly invoked. It verifies SHA-256 before installation and refuses an
invalid existing model. The runtime loads verified bytes so a path replacement
cannot substitute unchecked bytes between verification and model construction.
The `--model` option relocates this same pinned asset; new algorithms should be
added through a replacement backend, with their own asset policy.

Asset source and reference:

- [MediaPipe Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Model overview and model cards](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index)
- [Version-1 model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- Model SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

The model is not committed; obtain it from its original publisher and consult the
linked model cards and upstream licensing before redistribution.

The repository contains source code, tests, setup tools and documentation.
Virtual environments, downloaded models, diagnostic images, private evaluation
data, credentials and local agent configuration are excluded. The setup commands
run model retrieval explicitly; normal camera processing has no network dependency.

## Verify milestone 3

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_model.py
.venv/bin/python tools/smoke_tracking.py
.venv/bin/python tools/smoke_motion.py
.venv/bin/python tools/smoke_motion.py --json
.venv/bin/python -m pip check
```

The automated tests cover malformed frames, NaN/invalid landmarks and scores,
timestamps, quality failures, multiple faces, entering/leaving faces, model and
camera errors, cancellation, cleanup failures, hung workers, partial IPC, crashes,
unexpected success results and opt-in image permissions. The smoke helper runs
the real model on synthetic blank pixels without opening a webcam. The tracking
demo prints scalar JSON for synthetic still-face → loss → re-entry → appearance
substitution scenarios, exposing individual signal values and reset behavior.

Tracking tests cover stillness, gradual translation/scale/roll, eye/iris changes,
landmark jitter, sudden geometry/appearance substitution, positive brightness/
contrast changes, short frame gaps, expiry without new frames, bounded history,
immutable copies and cleanup on every terminal path. A re-entering face must show
ACQUIRING with a new track number, not resume STABLE with earlier history.

The motion demo generates still, small 3D rotation, moving plane, shared camera
motion and textureless-background scenarios offline. Only the generated 3D
rotation earns weak positive motion evidence; every complete session remains
UNCERTAIN with confidence zero. Tests also cover projective scene movement,
pose noise/unavailability, bounded caches, reset paths and strict motion IPC.
Generated geometry tests do not establish physical webcam sensitivity or attack
discrimination.

Verified here: 79 automated tests pass; the model initializes, performs blank-frame
inference and closes; missing-camera capture returns UNCERTAIN with confidence 0
and exit 2. No `/dev/video*` device is visible in this execution environment and
Tk is unavailable, so webcam and GNOME preview behavior need manual verification.
These checks do not measure presentation-attack performance.

## Privacy and known limits

Camera frames and geometry are transient memory by default. The tracker retains
bounded landmark samples, one geometry anchor and small appearance features.
The motion analyzer retains one reduced grayscale environment frame (at most
320 × 240), with the detected face and surrounding margin blacked out, plus its
mask and bounded scalar pose/transform histories. No earlier full camera frame
is retained. Loss, poor quality, expiry, rejection, cancellation and finish release
all these buffers. Only a clean finish may keep historical scalar motion diagnostics.
Public JSON excludes
frames, landmarks and embeddings. Native worker stderr is suppressed, and core
dumps are disabled in the worker. Optional preview sends an annotated frame through
the private child pipe into the local window; it does not print or save the image.

Only this explicit debug option retains raw biometric images:

```bash
.venv/bin/python -m facepass --save-diagnostic-frames /tmp/facepass-debug-new
```

It prints a prominent warning, requires a new directory with private POSIX
permissions, saves at most one PNG per second and caps retention at 30 images per
run. The developer is responsible for deleting retained files. Shared/non-POSIX
mounts that cannot enforce permissions are refused; this project's media mount may
fall into that category. Never use diagnostic retention in an authentication daemon.

Quality thresholds are initial engineering gates, not calibrated security policy.
Glasses, occlusion, severe pitch, backlighting, dark skin under poor exposure, small
faces and a camera's sharpening/noise can affect quality or landmark estimates.
Occlusion is explicitly unmeasured; calibrated landmark confidence is unavailable.
The predicted landmark z coordinate is not a measured depth signal. A face on a
photo or a screen can produce excellent landmarks and pass these quality and
tracking checks. A visually similar replacement face, mask, print or coherent
replay can evade continuity heuristics. Rapid motion, directional lighting changes
and landmark noise can also cause conservative UNCERTAIN results. Hardware tuning,
physical face-switch testing and motion/attack evaluation are outstanding.

Head pose uses a generic six-point 3D template, approximate camera intrinsics and
no calibrated lens distortion. It does not measure facial depth. The synthetic
tests recover subdegree rotations of that same template; real anatomical variation,
camera noise and landmark jitter can exceed these tiny changes. A coherent replay
can contain real 3D-looking movement and gain both weak scores. Moving prints,
background parallax, rolling shutter, reflections and autofocus can confound the
planar/background comparison. Little texture or no usable pose reduces evidence.
A still person or motion explained by a plane is never automatically labeled SPOOF.

Default tracking uses at least three consecutive usable frames spanning at least
150 ms. A gap over 350 ms clears temporal history; a session frame gap over one
second remains a terminal capture failure. Strong position/scale/shape/appearance
discontinuities and a camera resolution change terminate the attempt. These are
initial continuity gates, not calibrated security thresholds. Appearance evidence
is unavailable when the aligned skin region has too little visible texture;
geometric continuity may still become STABLE with its score capped at 0.5.

See [the milestone 3 report](docs/milestone-3-report.md) for implementation,
assumptions, testing and known failures. Blink detection begins only after new
direction.

Read [the architecture](docs/architecture.md), [threat model](docs/threat-model.md),
[manual evaluation plan](docs/evaluation.md) and [milestone roadmap](docs/milestones.md)
before continuing. Software-only RGB PAD cannot reliably provide IR/depth biometric
security. No unlock integration is implemented or appropriate at this stage.

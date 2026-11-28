# FacePass

Local RGB-camera research, with optional experimental face unlock for Fedora KDE.
Camera processing and enrolled face templates stay on your laptop.

## Current progress

**Version 0.14.0 · face enrollment, identity matching, randomized blink/head-turn challenges,
root authentication broker, native PAM module, visible KDE FacePass button, offline installer and rollback.**

Open **FacePass Setup** to install the system component and enroll your face. The
installer requires the laptop administrator password; enrollment requires you at
the camera. After both enrollment and a fresh verification pass, lock your screen
and select **FacePass**, or submit an empty password at Plasma Login. Entering your
normal password continues through Fedora's existing password stack.

See [face-unlock setup, limits and recovery](docs/face-unlock.md). RGB face unlock
is experimental: the implementation is tested, but biometric error rates and
resistance to sophisticated presentation attacks have not been established.

Milestones 1–9 implement local webcam capture, face landmarks, short-term face
continuity, experimental head-motion/geometry evidence, bilateral blink sequences
and relative iris movement, multi-region illumination response, conservative
replay/print artifact diagnostics, a bounded dependence-aware temporal consistency
summary and adaptive observation. **The research CLI and research daemon still
never return LIVE or grant authentication.** The separate system component uses
SFace identity templates and an active randomized challenge; it does not turn
research diagnostics into an authentication decision. Calibrated passive fusion
and production-grade RGB attack resistance remain unestablished.

| Milestone | Delivered | Validation still needed |
| --- | --- | --- |
| 1 — Camera and landmarks | V4L2 capture, local face mesh, quality gates, supervised diagnostic harness | Physical webcam capture and GNOME/KDE preview |
| 2 — Stable multi-frame tracking | One-face continuity, immutable bounded history, discontinuity refusal and reset handling | Physical face-switch, loss and reacquisition scenarios |
| 3 — Head micro-motion | Approximate pose, small landmark movement, planar fit, background compensation and two weak evidence producers | Physical motion sensitivity and print/display/replay evaluation |
| 4 — Blink sequences | Bilateral eye aspect ratio, timed closure/reopening, noise checks, sustained-closure state and reset handling | Physical blink sensitivity, glasses/occlusion and replay evaluation |
| 5 — Gaze movement | Relative iris coordinates, bilateral temporal motion, head-rotation confounder gate, eye resolution checks and resets | Physical eye-movement sensitivity, glasses/occlusion, head coupling and replay evaluation |
| 6 — Illumination | Seven facial patches and four outside-face ambient tiles, temporal RGB/shading response, common exposure/white-balance removal and motion gates | Physical lighting/reflectance sensitivity, camera processing and print/display/replay evaluation |
| 7 — Replay/print artifacts | Five face and four outside-face texture patches; periodic, chroma-grid, banding, sharpening, resampling, glare and edge diagnostics with persistence/motion gates | Physical camera/skin/fabric/room false positives and diverse print/display/replay detection rates |
| 8 — Temporal consistency and fusion | Bounded cross-frame support, explicit landmark/photometric/replay dependence groups, overlap/agreement/conflict diagnostics and strict scalar IPC validation | Physical cross-signal calibration, held-out attack rates and a validated policy |
| 9 — Adaptive observation | Minimum observation window, conflict-aware early completion, one bounded evidence extension, hard timeout and disabled challenge state | Physical usability, held-out completion/nonresponse rates and policy calibration |

The software has been checked on Fedora 44, x86_64 and Python 3.14.8. The current
session reports KDE Wayland; Tk display initialization, a hidden synthetic-image
preview and cleanup passed. The installed application can access two camera nodes
outside the workspace sandbox. Physical capture evaluation across lighting, blur
and presentation attacks remains outstanding. Automated environment fixtures also cover
GNOME and headless sessions. Passing these checks does not establish attack
resistance. See the [Fedora KDE guide](docs/fedora-kde.md).

The remaining software roadmap is now implemented in one pass. Physical evaluation
and calibration remain required before treating RGB authentication as production-grade.

## Evaluation, performance and daemon

| Milestone | Delivered software | Validation still needed |
| --- | --- | --- |
| 10 — Attack evaluation | Private scalar attempt recorder, strict JSONL validation, stratified reports, rates and intervals, held-out leakage checks | Real users, cameras, rooms, glasses, lighting, prints/displays/replays and repeated-attempt dependence |
| 11 — CPU and latency | Bounded capture profiles, offline full-pipeline/real-model benchmarks, optional function profiles, consistent extension deadlines | Physical laptop CPU, thermal behavior, latency and usability under load |
| 12 — Daemon interface | Version-2 same-user Unix socket with observe/status/cancel, one supervised worker, disconnect/shutdown cleanup and a CLI client | Physical webcam operation; independent review and calibration before any authentication integration |

Each milestone includes tests, an implementation report, assumptions and known
failure cases. Physical capture checks should begin as soon as a webcam is available,
and attack evaluation must inform future calibration. LIVE stays disabled until
diverse evidence, quality and fusion policy are validated.
The separate experimental authenticator now integrates Fedora KDE's lock screen
and optionally Plasma Login/SDDM. GNOME/GDM face unlock is not implemented.

Milestone reports: [1](docs/milestone-1-report.md),
[2](docs/milestone-2-report.md), [3](docs/milestone-3-report.md),
[4](docs/milestone-4-report.md), [5](docs/milestone-5-report.md),
[6](docs/milestone-6-report.md), [7](docs/milestone-7-report.md),
[8](docs/milestone-8-report.md), [9](docs/milestone-9-report.md),
[10–12](docs/completion-report.md).

After installing the project, the new commands are also available as
`facepass-evaluate`, `facepass-daemon`, and `facepass-client`. Module invocation
works directly in the existing development environment:

```bash
# Bounded end-to-end scalar timings; no images retained. Capture exits 2.
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --profile --json

# Offline detector cost; optionally include verified real-model blank inference.
.venv/bin/python tools/profile_pipeline.py --frames 60 --model models/face_landmarker.task

# One explicitly labelled physical attempt; choose anonymous codes for metadata.
.venv/bin/python -m facepass.evaluation capture --output /tmp/facepass-attempts.jsonl \
  --label bona_fide --scenario still --split development \
  --participant p001 --room room001 --media none --device-code cam001
.venv/bin/python -m facepass.evaluation report /tmp/facepass-attempts.jsonl

# Start in one terminal; it remains idle until an observe request.
.venv/bin/python -m facepass.daemon
# Run in a second terminal using the same runtime environment.
.venv/bin/python -m facepass.client status
.venv/bin/python -m facepass.client observe --request-id trial001 --duration 5
.venv/bin/python -m facepass.client cancel trial001
```

Observation may finish early or receive one extension of at most two seconds by
default. `--fixed-window` disables both. `--extension-duration 0..5` sets the one-time
extension limit; the worker and supervisor enforce the same bound. The maximum
active capture is 35 seconds, plus bounded startup and cleanup supervision.

Evaluation reports keep physical/synthetic sources, versions, policies,
configurations and splits in separate cohorts. Nonresponses and quality failures
remain visible. Every current capture is non-success, so zero accepted attacks
cannot be interpreted as validated security. Logs contain scalar outcome summaries,
not frames or landmarks, and require a private 0600 POSIX file. Keep them private.
See [evaluation tooling](docs/evaluation-tooling.md) and [daemon protocol](docs/daemon-protocol.md).

## Blur, low quality and dim faces

The default `tolerant` capture profile accepts moderate blur, smaller resolved
faces and dim facial detail. It adjusts the sharpness check for brightness instead
of applying the original absolute brightness cutoff to every face. A bounded
full-frame gamma pass helps landmark detection in dim scenes. Quality, tracking
appearance, illumination and replay detectors still receive the original pixels.
No sharpening or generated detail is used.

```bash
facepass --preview
# Compare with the original capture gates:
facepass --preview --quality-profile strict
```

Accepted compromises appear as `moderate_blur`, `low_luminance`, `limited_contrast`
or `small_face` warnings, including in the preview and final diagnostics. These
warnings allow tracking to continue. Missing detail, near-black/washed-out capture,
severe blur, clipping, invalid orientation and multiple faces still prevent usable
capture. Individual detectors retain their own resolution and evidence checks.
Tolerance is experimental; synthetic exposure/blur/compression regressions do not
establish performance for skin tones or physical cameras. See the
[quality profile details](docs/capture-quality.md).

## Implemented capabilities

- An ordinary-user `/dev/videoN` V4L2 capture adapter, requesting 640 × 480 at
  12 processed frames per second by default. Device settings are best effort.
- Local MediaPipe Face Landmarker inference with 478 landmarks per detected face,
  including eyelid contours and iris geometry. Up to four faces are detected afresh
  on each frame. Two or more detected faces terminate the attempt as ambiguous.
- Experimental exposure-aware gates for face size, clipping, blur, contrast,
  exposure and rough orientation; tolerant capture is the default, with strict
  comparison gates available through `--quality-profile strict`.
  Zero faces and poor quality produce INSUFFICIENT_QUALITY; camera/model failures
  produce UNCERTAIN. Quality never supplies liveness evidence.
- A bounded session API with transient geometry and scalar diagnostics. Head
  motion, facial geometry, blinking, gaze, illumination and replay-artifact analysis
  have weak experimental evidence producers. Milestone 8 adds bounded temporal
  consistency and an explicit dependence-aware fusion summary while keeping global
  confidence at zero.
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
- A separate blink analyzer measuring both eye contours in pixel coordinates.
  It requires an open baseline, at least two consecutive closed samples spanning
  60 ms, then confirmed reopening within a 650-ms sequence limit. Single-frame
  collapse, asymmetric eyes, sustained closure and unreliable geometry do not
  earn blink credit. No blink stays neutral; blinking is optional.
- Blink evidence capped at confidence 0.65 and reliability 0.35, expiring after
  two seconds. Gaps over 220 ms clear blink state even if face tracking is still
  stable. Blink and motion share the landmark model and are correlated evidence.
- A separate relative-iris analyzer with a replaceable evidence function. Eye-local
  coordinates remove image translation, scale and roll; six usable samples spanning
  450 ms, bilateral motion agreement, a two-pixel displacement floor and little
  head rotation are required for weak positive evidence. Still eyes remain neutral;
  no gaze direction or deliberate movement is required.
- Gaze confidence capped at 0.62 and reliability at 0.30. Small/closed/unreliable
  eyes, poor iris geometry, unusable pose and gaps over 220 ms clear gaze history
  without rejecting an otherwise usable face. This is relative eye movement, not
  calibrated gaze direction, screen position or an eye-visibility guarantee.
- A separate illumination analyzer samples forehead, nose, both cheeks, both
  subocular socket patches and chin, against four ambient tiles outside an expanded
  face hull. It measures ambient RGB balance and regional shading, then compares
  spatial RGB response across time after removing common per-channel gains as
  possible exposure/white-balance effects. Uniform brightening stays neutral.
- Illumination confidence capped at 0.58 and reliability at 0.20. Eight samples
  spanning 600 ms, coherent spatial response across several face regions and little
  head/scene movement are required. Static lighting, unavailable evidence and
  response mismatch never automatically mean SPOOF. No lighting challenge is required.
  Direction is an image-plane response proxy; room-light direction, Kelvin color
  temperature and facial depth are not calibrated or measured.
- A replay-artifact analyzer compares five interior face texture patches with four
  outside-face controls. It reports periodic/halftone concentration, chroma grids,
  row/column banding, sharpening, lag periodicity as a resampling proxy, glare and
  axis-aligned edge concentration. These camera/content-dependent measurements do
  not identify a physical display or print by themselves.
- Weak suspicion requires at least two artifact families, three affected facial
  regions and 75% temporal persistence over six samples/450 ms, with usable
  background motion and no whole-scene contamination. It is capped at liveness
  confidence 0.35/reliability 0.25 and never directly returns SPOOF. When no
  persistent artifact is observed, the signal stays neutral; missing artifacts
  are never positive liveness evidence.
- Temporal consistency retains at most 64 scalar snapshots over two seconds. The
  landmark/pose, photometric scene and replay texture groups are explicit; the four
  landmark-derived signals contribute at most one group. Cross-group overlap,
  stability, missing groups, replay conflicts and a dependence penalty are exposed
  for evaluation. Two groups can earn only a weak temporal signal (0.56/0.18); this
  diagnostic never becomes LIVE or SPOOF and never hides individual scores.
- Adaptive observation uses a 1.5-second minimum window, finishes early only for
  stable conflict-free two-group support, requests at most one two-second extension
  for partial evidence, and then applies the hard timeout. The policy is diagnostic
  and challenge issuance is explicitly disabled; it cannot change the global
  UNCERTAIN/INSUFFICIENT_QUALITY decision.
- A finite worker supervised by the terminal harness, with a startup limit,
  capture/model stall detection, JSON validation, crash detection and cancellation.
- Optional annotated preview and explicit, disabled-by-default diagnostic image
  retention. No runtime downloads, network calls or identity recognition.

## Use on Fedora GNOME or KDE Plasma

Run as your normal user from GNOME Terminal or KDE's Konsole in the project
directory. For a fresh clone, first follow the environment setup below; `.venv` and the model are not
included in the repository. Start with:

```bash
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5
```

The doctor command reports the desktop, session type and preview prerequisites
without opening the webcam or a window. `preview.status = ready_to_try` means Tk
and a configured X11/XWayland display are present; it does not verify a display
connection. Select a readable/writable capture
node from its camera list. Some webcams expose a separate metadata node which
cannot deliver video; try the other capture node if opening/reading fails. A
missing or unavailable camera safely ends the attempt.

Every capture run ends with the final decision and individual signal confidence,
reliability and availability. `--json` emits structured scalar updates and the
final result, including tracking state, current session-local track number, history
count, reset count/reason, approximate pose, scalar motion diagnostics and blink
phase, eye measurements, completed/rejected sequences and timing. Gaze diagnostics
include current relative iris positions, window span/coherence, bilateral agreement,
head-rotation span and whether independent motion was observed.
Illumination diagnostics include ambient luminance/RGB balance, regional log contrasts,
common gain/color shift, window counts, spatial response directions, coherence,
regional residual, color error, motion suppression and mismatch/consistency flags.
Replay diagnostics include seven family scores, outside-face artifact level,
affected regions, family count, persistence, temporal variation and motion suppression.
Temporal diagnostics include bounded window size/duration, group support, overlap,
agreement, stability, conflicts and missing groups. Fusion diagnostics include group
scores, coverage, dependence penalty, adjusted support and explicit disabled candidate
flags. Historical diagnostics are marked unavailable for live evidence on clean finish.
Observation diagnostics include elapsed time, minimum/base/effective limits, extension
count, quality/stable-frame coverage, readiness, completion and disabled challenge state.
Confidence 0.5 denotes neutral evidence; none of these heuristic signal scores
is a measured liveness probability. Overall liveness confidence stays zero.
On a clean finish, motion, blink, gaze, illumination, replay and temporal signal
values are marked historical, with live signal availability false and reliability
zero; fusion is marked historical for diagnostics. Images, geometry and detector caches are
released. Failure, cancellation, quality loss and staleness clear those values.

The harness exits with code **2** for every capture result at this milestone,
including a clean quality capture. `--doctor`, the unit tests, and the model setup
helper may exit 0 for their respective tooling success; they are not authentication
entry points. Ctrl+C cancels capture; no pipeline runs after the harness exits.

Optional in-memory landmarks preview:

```bash
.venv/bin/python -m facepass --device /dev/video0 --preview
```

Preview uses Tk rather than OpenCV's Qt window to avoid coupling the camera engine
to Qt platform plugins. Fedora's [python3-tkinter package](https://packages.fedoraproject.org/pkgs/python3.14/python3-tkinter/)
and a desktop display are required for the system Python. On GNOME or Plasma
Wayland, Tk uses [XWayland](https://packages.fedoraproject.org/pkgs/xorg-x11-server-Xwayland/xorg-x11-server-Xwayland/).
For the optional preview on Fedora's system Python:

```bash
sudo dnf install python3-tkinter xorg-x11-server-Xwayland
```

Launch from a terminal inside the logged-in desktop so it inherits `DISPLAY`.
The [KDE setup and troubleshooting guide](docs/fedora-kde.md) covers Plasma Wayland.
If Tk or the display is unavailable, run terminal mode; the preview does not silently fall back to recording.
Close the window or press Escape to cancel. Terminal mode needs no display server.

This prototype reads V4L2 devices directly. It is not a Flatpak or a PipeWire camera
portal client, and does not run at GDM, SDDM or either desktop's lock screen. Keep
SELinux enforcing and use existing desktop camera permissions. Do not run it as root, change device
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

## Verify the complete software

The single verification command runs unit/integration tests, dependency checks,
real-model blank inference and every generated scenario helper. It uses temporary
local sockets and never opens a webcam:

```bash
.venv/bin/python tools/verify_project.py
```

On a checkout without a downloaded model, use `--skip-model` explicitly, then
obtain the pinned asset with the setup helper before running the full checks.
The GitHub Actions workflow runs these same checks in a clean Python environment.
An execution sandbox must permit local Unix sockets for daemon integration tests.

Individual checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_model.py
.venv/bin/python tools/smoke_tracking.py
.venv/bin/python tools/smoke_motion.py
.venv/bin/python tools/smoke_motion.py --json
.venv/bin/python tools/smoke_blink.py
.venv/bin/python tools/smoke_blink.py --json
.venv/bin/python tools/smoke_gaze.py
.venv/bin/python tools/smoke_gaze.py --json
.venv/bin/python tools/smoke_illumination.py
.venv/bin/python tools/smoke_illumination.py --json
.venv/bin/python tools/smoke_replay.py
.venv/bin/python tools/smoke_replay.py --json
.venv/bin/python tools/smoke_temporal.py
.venv/bin/python tools/smoke_temporal.py --json
.venv/bin/python tools/smoke_observation.py
.venv/bin/python tools/smoke_observation.py --json
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

The blink demo generates a bilateral sequence, open stillness, sustained closure,
single-frame noise, asymmetric eyes, small eyes, a sampling gap and an identical
replay pattern. Only the completed sequence and its identical replay earn the weak
blink score. Tests cover hysteresis, minimum sampled duration, credit expiry,
incomplete finish, all reset paths and strict scalar IPC. These fixtures alter
landmarks over static generated pixels; they do not measure physical eyelid detection
or print/replay performance.

The gaze demo generates stillness, horizontal/vertical movement, a fixation change,
scene movement, head rotation, one-eye motion, single-frame noise, small iris,
partial eye failure, closure, a gap and an identical replay pattern. Coherent
bilateral movement and its identical replay can earn the same weak score. Tests
cover eye-coordinate invariance, noise/pose gates, bounded scalar history, freshness,
cleanup and strict gaze IPC. These are generated geometry checks, not physical
gaze accuracy or attack evaluation.

The illumination demo generates fifteen spatial-lighting, common-gain, color,
planar-response, noise, missing-context, motion, gap and identical-replay cases.
Only qualifying multi-region response and its identical replay earn the weak score.
These are controlled pixel-gain fixtures, not physical lighting or PAD evaluation.
Tests cover outside-face masks, RGB conversion, exposure/WB removal, conservative
neutral behavior, bounded private summaries, polling expiry, all resets and strict IPC.

The replay demo generates thirteen natural, compound display/print, camera-wide,
background-only, glare, sharpening, low-texture, motion, gap, transient and
origin-ambiguous cases. Generated display-grid and halftone patterns receive weak
suspicion; global sensor banding and a background grid stay neutral. The same
pattern labeled as bona fide is also flagged, explicitly demonstrating that the
fixture cannot establish origin or physical attack accuracy.

Verified here: the automated suite covers the detector pipeline and new evaluation/daemon tools; the model initializes, performs blank-frame
inference and closes; missing-camera capture returns UNCERTAIN with confidence 0
and exit 2. No `/dev/video*` device is visible in this execution environment and
physical capture is unavailable, so webcam and GNOME/KDE preview behavior need manual verification.
These checks do not measure presentation-attack performance.

## Privacy and known limits

Camera frames and geometry are transient memory by default. The tracker retains
bounded landmark samples, one geometry anchor and small appearance features.
The motion analyzer retains one reduced grayscale environment frame (at most
320 × 240), with the detected face and surrounding margin blacked out, plus its
mask and bounded scalar pose/transform histories. No earlier full camera frame
is retained. Loss, poor quality, expiry, rejection, cancellation and finish release
all these buffers. The blink analyzer retains only current scalar eye measurements,
two preceding eye widths, phase timers and counters; it stores no additional frames
or raw eye/landmark series. The gaze analyzer keeps at most 64 scalar eye-position/
pose samples over two seconds and preceding eye/iris geometry; no additional
images or full landmarks. The illumination analyzer creates temporary masks and
a reduced color frame (at most 320 × 240), linearizing at most 512 sampled pixels
per region. Its private history retains only seven facial and four ambient RGB
medians, timestamps and motion context, at most 64 samples over two seconds.
It retains no additional image or mask. Only a clean finish may keep historical
scalar motion, blink, gaze, illumination, replay, temporal and fusion diagnostics.
The temporal analyzer stores only bounded scalar group supports and timestamps. The replay analyzer
temporarily resizes nine equal-sized patches to 32 × 32 and retains only seven
bounded feature scores per region, motion context and timestamps (64 samples/two
seconds maximum); no patch, spectrum, frame or mask is retained.
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

Blink thresholds (open EAR ≥ 0.23, closed EAR ≤ 0.15, eye width ≥ 18 pixels) are
uncalibrated and not personalized. Glasses, occlusion, eye anatomy, perspective and
landmark hallucination can cause missed or false sequences. No calibrated eye
visibility measure is available. At the default 12 processed frames per second,
fast natural blinks may not provide two closed observations; no interpolation or
threshold lowering compensates for missing samples. A prerecorded blink can gain
exactly the same weak score as a live blink. Blink count is neither identity nor
liveness, and no blink is never automatic SPOOF.

Gaze geometry gates require eye width ≥ 28 px, both iris diameters ≥ 6 px and open
eyes. Relative displacement must exceed both 0.035 eye widths and two pixels, with
coherence/agreement ≥ 0.70 and head rotation span ≤ 1.5°. These choices are
uncalibrated; tiny or nonlinear natural gaze, rapid saccades and genuine movement
during head rotation can remain neutral/unavailable. Glasses/reflections/occlusion
can corrupt plausible landmarks without being recognized. Partial eye failure
reduces gaze reliability to zero; this version does not detect glasses or estimate
calibrated eye visibility. Replayed iris motion can earn the same score. Missing
gaze never forces a challenge or an automatic SPOOF decision.

Illumination uses an approximate inverse-sRGB transfer, fixed landmark patches and
image-plane spatial fits. Skin/facial hair, glasses, specular highlights, colored
walls, mixed light, occlusion, camera tone mapping and predicted landmarks can
confound the measurements. Ambient RGB balance is not illuminant chromaticity or
Kelvin temperature; response direction is not the actual emitter direction.
Common-gain removal cannot distinguish true uniform room changes from camera
exposure/WB adjustments, and intentionally leaves those changes neutral. No camera
controls are read or changed. Regional residual is not measured 3D geometry or a
validated print discriminator. A print can respond to room light and a replay can
match every gate; identical generated replay receives the same weak score.
Missing pose/background regions, clipped/dark patches, abrupt common color/gain
jumps or gaps over 350 ms clear illumination history. Head/scene movement or missing
background-flow context suppresses positive evidence without automatic rejection.

Replay thresholds are fixed engineering gates, not calibrated classifier outputs.
Skin, facial hair, glasses, fabric, room patterns, sensor demosaicing, rolling
shutter, sharpening, compression and resizing can imitate display/print artifacts.
Modern high-resolution displays and quality prints can show none of them. Outside-face
controls reduce camera-wide confounding but cannot identify a device boundary or
guarantee comparable scene texture. Motion blur suppresses suspicion conservatively.
The detector requires persistent agreement from two families; those measurements
remain correlated and do not become independent votes. Its generated fixtures are
mathematical patterns rather than photographed replay media.

Default tracking uses at least three consecutive usable frames spanning at least
150 ms. A gap over 350 ms clears temporal history; a session frame gap over one
second remains a terminal capture failure. Strong position/scale/shape/appearance
discontinuities and a camera resolution change terminate the attempt. These are
initial continuity gates, not calibrated security thresholds. Appearance evidence
is unavailable when the aligned skin region has too little visible texture;
geometric continuity may still become STABLE with its score capped at 0.5.

See [the milestone 8 report](docs/milestone-8-report.md) for implementation,
assumptions, testing and known failures. Physical calibration and authentication
policy remain future work.

Read [the architecture](docs/architecture.md), [threat model](docs/threat-model.md),
[manual evaluation plan](docs/evaluation.md) and [milestone roadmap](docs/milestones.md)
before continuing. Software-only RGB PAD cannot reliably provide IR/depth biometric
security. No unlock integration is implemented or appropriate at this stage.

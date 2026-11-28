# Milestone 5 — optional relative iris movement

Version 0.5.0 adds a separate gaze evidence producer to the standalone Fedora GNOME
harness. It estimates iris movement relative to the eyes over time, with bilateral,
resolution, noise and head-rotation gates. It does not estimate calibrated gaze
angles or where a person is looking on a screen. No deliberate gaze direction or
movement is required; missing/still gaze is neutral or unavailable, never automatic
SPOOF. Overall confidence remains zero, LIVE stays disabled and camera runs exit 2.
No PAM, GDM, GNOME or system authentication files were changed. Development stops
here before milestone 6 (illumination estimation).

## Implementation and assumptions

`gaze.py` contains `GazeAnalyzer`; `gaze_evidence.py` contains a replaceable evidence
function. Each processed STABLE, quality-accepted track sample supplies MediaPipe
eye and iris geometry. Four ordered rim points per iris provide a centroid. Pixel
coordinates express horizontal and perpendicular centroid displacement from the
image-left eye corner, divided by eye width. Both components use eye-width units,
not image axes or eyelid height; uniform image translation/scale/roll cancel.
Perspective, eyelid-corner noise and anatomy still affect these ratios.

Geometry checks require open eyes, enough pixels, in-frame and plausible contours,
convex rings, consistent opposite diameters/centers/directions and a centroid in
the eye aperture. Ring landmarks may be predicted behind an eyelid; accepted
geometry does not establish physical visibility. If either eye fails, gaze history
is cleared and reliability becomes zero while an otherwise usable face can remain
STABLE. Glasses, occlusion and calibrated landmark confidence are not directly measured.

The generic approximate head-pose estimator supplies yaw/pitch/roll, using the same
uncalibrated template/intrinsics as milestone 3. Missing/nonfrontal pose clears the
window; non-finite/native failures terminate the session without success. Gaze and
motion run as separate producers but share model/pose dependence.

| Initial engineering gate | Value |
| --- | --- |
| Eye openness | Both EAR ≥ 0.23 |
| Eye width | Each ≥ 28 px |
| Iris size | Both opposite diagonals ≥ 6 px; average diameter/eye width 0.15–0.5 |
| Eye/iris geometry continuity | Absolute logarithmic geometry change ≤ 0.2 per sample |
| Iris position continuity | Each coordinate step ≤ 0.12 eye widths |
| Pose range | Absolute yaw ≤ 25°, absolute pitch ≤ 20° |
| Evidence window | ≥ six samples spanning ≥ 450 ms |
| History limit | ≤ 64 scalar samples over ≤ two seconds |
| Sample freshness | ≤ 220 ms; larger gaps reset state |
| Relative motion | Both eyes' robust displacement ≥ 0.035 widths and ≥ two pixels at the current smaller eye width |
| Trend coherence / bilateral agreement | Each ≥ 0.70 |
| Head rotation span for positive evidence | ≤ 1.5° in every approximate pose component |

Robust displacement compares the medians of the first and last three samples in
each eye. Agreement combines nonnegative direction cosine and smaller/larger
displacement magnitude. A joint linear trend supplies coherence. Percentile spans
are diagnostic only; a large span alone cannot earn credit. Sustained fixation
changes and gradual movement can qualify, but nonlinear natural movement can remain
neutral. Whole-frame motion cancels in the coordinate system; substantial head
rotation conservatively suppresses positives rather than claiming 3D compensation.

| Observation | Confidence | Reliability | Availability |
| --- | --- | --- | --- |
| Short/unreliable eye or pose observation | 0.50 | 0 | False |
| Usable eyes, no qualifying independent motion | 0.50 | 0.10 | True after the minimum window |
| Head rotation confounds the usable window | 0.50 | 0.05 | True after the minimum window |
| Qualifying bilateral relative iris motion | 0.62 | 0.30 | True |

These values are fixed, uncalibrated engineering scores, not liveness probabilities.
They are recomputed from the bounded window; old movement expires during stillness.
`current_result()` trims old samples and enforces freshness even between frames.
No interpolation, threshold lowering, challenge or authentication shortcut was added.

Private memory holds timestamp, two x/y pairs and three pose angles per sample,
plus current scalars and preceding small geometry/position arrays. It holds no new
image/full-landmark cache. Track change, loss, poor quality, stale capture, jump,
multiple faces, exception, failure and cancellation clear state. A clean fresh
finish retains only historical scalar diagnostics, with evidence unavailable and
reliability zero; buffers are released. Cleanup failure clears that summary too.

Terminal/preview output shows gaze confidence and availability; final output shows
all individual scores/reliability. JSON includes current scalar positions, eye/iris
resolution, window counts/span/displacement/coherence/agreement, head span and flags.
Strict worker IPC checks known fields, finite bounds/types, score caps, stable quality
context and independent-motion conditions. Raw trajectories, landmark arrays,
images and embeddings are not serialized.

The iris rim indices were checked against [official MediaPipe contour definitions](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py).
[MediaPipe's iris documentation](https://chuoling.github.io/mediapipe/solutions/iris.html)
distinguishes iris landmarks from estimating the location a person looks at. This
engineering motion proxy does not inherit upstream accuracy claims or measured depth.

## How to test on Fedora GNOME

From the project directory with the existing venv and verified model:

```bash
.venv/bin/python tools/smoke_gaze.py
.venv/bin/python tools/smoke_gaze.py --json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Use a capture node reported by doctor and your ordinary Fedora desktop user. For
mesh preview, omit `--json` and add `--preview` when Fedora's `python3-tkinter` and
a GNOME display are available. Naturally look at the laptop; inspect gaze availability,
current ratios, head suppression and the final non-success. Raw camera retention
stays disabled by default. Follow [evaluation.md](evaluation.md) for physical cases.

The offline tool generates thirteen scenarios without opening a webcam:

| Generated scenario | Confidence / reliability | Expected |
| --- | --- | --- |
| Still open eyes | 0.50 / 0.10 | Neutral |
| Bilateral horizontal movement | 0.62 / 0.30 | Weak evidence |
| Bilateral vertical movement | 0.62 / 0.30 | Weak evidence |
| Confirmed fixation step | 0.62 / 0.30 | Weak evidence |
| Whole-scene translation | 0.50 / 0.10 | Relative position unchanged |
| Iris changes plus head rotation | 0.50 / 0.05 | Head-confounded, neutral |
| One eye moving | 0.50 / 0.10 | Bilateral condition unmet |
| Single-frame spike | 0.50 / 0.10 | Noise not rewarded |
| Small iris | 0.50 / 0 | Unavailable |
| Partial eye failure | 0.50 / 0 | Unavailable |
| Eye closure | 0.50 / 0 | Unavailable |
| Sequence crossing a 300-ms gap | 0.50 / 0 | Reset; final window too short |
| Identical replay-pattern geometry | 0.62 / 0.30 | Same weak score, origin undetermined |

These fixtures alter idealized landmarks over static generated pixels, not actual
eye imagery. The replay row repeats the same geometry; it is not a physical display
test. A complete generated session finishes UNCERTAIN, confidence 0, gaze values
historical, evidence unavailable and retained tracking/scalar history zero. The
tool's exit 0 means engineering verification, not authentication.

## Verification performed

Environment: Fedora 44 Workstation, GNOME Wayland, x86_64, Python 3.14.7. Editable
metadata is 0.5.0. Existing MediaPipe, NumPy, OpenCV and the verified model were reused;
no new runtime dependency or system package was required.

| Check | Result |
| --- | --- |
| Full automated suite | 135 tests passed, including 32 new gaze/session tests |
| Generated gaze demo and JSON | Thirteen scenarios passed; terminal JSON passed strict validation |
| Existing blink demo | Passed; sequence/reset behavior and non-success preserved |
| Real model initialization, blank inference and cleanup | Passed; zero faces, no webcam opened |
| Dependency consistency | `pip check`: no broken requirements |
| Doctor | Milestone 5, authentication disabled, no visible camera, Tk unavailable |
| Missing-camera capture | UNCERTAIN, confidence 0, no retained history, exit 2 |
| Physical eyes, glasses, print/display/replay and GNOME preview | Not run: no camera visible; Tk unavailable |

New tests cover pixel-space coordinates, translation/scale/roll invariance, gradual
horizontal/vertical movement, confirmed steps, stillness, short/subpixel windows,
single spikes/jitter, unilateral/opposed motion, scene/head confounders, missing or
severe pose, closure/partial/small/malformed iris geometry, sudden jumps, all temporal
reset boundaries, bounded history, expiry during updates/polls, NaN/native failures,
historical finish, cleanup failure, inference completing after timeout, strict IPC
and unsupported LIVE. Existing capture/tracking/motion/blink/watchdog tests pass.

Review found that a short observation window at the default 12-fps cadence could
leave a NumPy boolean in diagnostics, which JSON cannot encode. The flag now uses
a Python boolean; a regression test validates every result at 12 and 30 fps,
including the interval before the 450-ms minimum is reached.

Polling also recomputes native trend math. Exceptions during either polling or
finish now terminate as `temporal_evidence_error`, release the backend and all
temporal buffers, and retain no historical evidence; regression tests cover both.

## Known limits and future work

- Tiny, fast, nonlinear or uneven natural eye motion can remain neutral/unavailable.
  The two-pixel floor can miss subtle movement; a step over 0.12 widths clears state.
  Low cadence can miss motion or break freshness; CPU/latency profiling is outstanding.
- Genuine eye movement during head turning is conservatively suppressed. Generic
  pose/intrinsics and perspective can misattribute movement or make it unavailable.
  Eye-local normalization is not calibrated eyeball/head compensation.
- Glasses, reflections, partial visibility and eye anatomy can corrupt plausible
  landmarks. Geometry gates cannot recognize every occlusion or false iris. A single
  bad eye makes this producer unavailable, without arbitrary spoof rejection.
- Prerecorded/generated eye motion or model hallucination on flat media can satisfy
  every gate. The producer does not distinguish the source; no physical attack
  false-acceptance/rejection or gaze sensitivity has been measured.
- Tracking does not establish identity. All implemented producers share measurements
  and require dependence-aware, empirically validated fusion before LIVE can exist.

Milestone 6 remains planned. Physical tests across users, cameras, rooms, glasses,
eye/head coupling and replay media must guide later calibration. No security or
IR/depth-equivalence claim follows from these synthetic checks.

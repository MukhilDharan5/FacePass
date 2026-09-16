# Milestone 3: passive head micro-motion and geometry diagnostics

Implemented for the existing Fedora GNOME standalone harness. This milestone adds
two experimental motion evidence producers and their temporal measurements.
**Every observation still returns UNCERTAIN or INSUFFICIENT_QUALITY, overall
confidence zero and capture exit code 2. LIVE, SPOOF and fusion remain disabled.**
No PAM, GDM, GNOME, sudo, login or system authentication configuration was changed.
Development stops here before milestone 4 (blink detection), as the supplied brief
requires after each milestone.

## What changed

- `head_pose.py`: approximate yaw/pitch/roll from six face landmarks, a generic
  engineering 3D template and OpenCV PnP. High reprojection error, implausible angles
  and invalid solutions reduce availability; non-finite/native failures fail closed.
- `motion.py`: bounded pose trends, landmark displacement, held-out planar-fit
  residuals and sparse background-relative motion over a shared temporal baseline.
  Background corners exclude the face and the artificial mask border, require
  forward/backward flow agreement, inlier agreement and spatial scene coverage.
  A simpler similarity scene transform is preferred unless projective motion
  clearly improves the fit.
- `motion_evidence.py`: separate, replaceable head-motion and geometry evidence
  functions. Both require STABLE quality tracking and at least six frames spanning
  450 ms. Stillness, incoherence, missing background texture, shared scene movement
  and an adequate planar explanation stay neutral. There is no required head turn,
  blink or active challenge.
- `session.py` / `results.py`: scalar-only motion diagnostics and cache resets on
  every continuity, quality, freshness and terminal boundary. A clean finish keeps
  only explicitly historical scalar diagnostics; both motion signals become
  unavailable with reliability zero. Errors and cancellation clear even that summary.
- `supervisor.py` / `cli.py`: strict scalar IPC validation and visible motion
  diagnostics in terminal, JSON and optional preview. Experimental score caps are
  validated; any LIVE result or nonzero overall confidence is rejected.
- `tools/smoke_motion.py`, `tools/synthetic_motion.py` and `tests/test_motion.py`:
  generated geometry/pixel scenarios, integration checks and fail-closed tests.
  Synthetic backends are tool/test fixtures, not camera CLI input or a success bypass.

Positive confidence is capped at 0.68. Head-motion reliability is at most 0.4;
geometry reliability is at most 0.35. These heuristic scores are uncalibrated and
cannot establish liveness. The two producers share landmarks, pose, planar fit and
background measurements; future fusion must account for this dependence.
The other five signal producers remain unavailable.

## Assumptions and interpretation

The existing trusted local V4L2 RGB camera, local pinned model, ordinary desktop
user and quality-gated one-face track assumptions remain. PnP uses a centered
principal point and focal length equal to the larger image dimension, with no
calibrated lens distortion or personalized anatomy. Neither that generic template
nor MediaPipe's predicted landmark z is measured depth.

The generated test recovers yaw 0.2°, pitch 0.1° and roll 0.05° for the same idealized
template and camera used by the estimator. A 0.05°-per-frame yaw sequence also
produces weak evidence. This demonstrates numerical sensitivity under ideal
inputs, not physical webcam accuracy or a proven ability to discriminate prints.
Real landmark noise can exceed these motions. No pose calibration, physical attack
evaluation or security threshold tuning was performed.

A linear-trend coherence check conservatively distinguishes generated steady
motion from unstructured jitter. Real nonlinear natural motion may remain neutral.
Lack of motion is missing evidence, never proof of spoofing. The 450-ms minimum is
an evidence window, not an authentication latency guarantee or early-success rule.

## Run and inspect

From the project directory in GNOME Terminal:

```bash
.venv/bin/python -m facepass --doctor
.venv/bin/python tools/smoke_motion.py
.venv/bin/python tools/smoke_motion.py --json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Select a usable device from `--doctor`; no camera is opened by doctor or the
synthetic demo. Omit `--json` for compact live terminal diagnostics. Add `--preview`
only when Fedora's `python3-tkinter` and a GNOME display are available; no system
package was installed by this milestone.

The offline demo prints:

| Generated scenario | Head confidence | Geometry confidence | Interpretation |
| --- | --- | --- | --- |
| Still | 0.500 | 0.500 | Neutral, not SPOOF |
| Subtle 3D rotation | 0.680 | 0.680 | Weak correlated evidence, not LIVE |
| Moving plane | 0.500 | 0.500 | Planar motion is not rewarded |
| Shared camera motion | 0.500 | 0.500 | Common scene motion is suppressed |
| Textureless background | 0.500 | 0.500 | Missing scene evidence reduces reliability |

Its complete session finishes UNCERTAIN, confidence 0.0, historical motion marked
true and retained landmark history zero. The synthetic tool exits 0 for engineering
verification; it is not an authentication entry point. All camera runs exit 2.

For physical testing, naturally look at the laptop without deliberately moving.
Inspect pose availability/error, rotation span, coherence, landmark displacement,
background availability/inlier fraction, common versus independent motion and
held-out planar error. Repeat stillness, glasses, lighting, moving prints/displays,
replays, camera movement and loss/failure cases in `evaluation.md`. Record scalar
results; raw frame retention remains explicitly opt-in and disabled by default.

## Verification performed

Environment: Fedora 44 Workstation, GNOME Wayland, x86_64, Python 3.14.7. Editable
project version is 0.3.0. Existing MediaPipe, NumPy and OpenCV dependencies and the
verified model were reused; no new dependency, model or system package was needed.

| Check | Result |
| --- | --- |
| Full unit suite | 79 tests passed, including 23 new motion/session tests |
| Generated motion demo and scalar JSON | Passed; confounders neutral, complete session non-success |
| Real model initialization, blank inference and cleanup | Passed; zero faces on generated blank pixels |
| Dependency consistency | `pip check`: no broken requirements |
| Doctor | Milestone 3, authentication disabled, no visible camera, Tk unavailable |
| Missing-camera terminal/JSON capture | UNCERTAIN, confidence 0, no retained motion/history, exit 2 |
| Physical faces, prints, displays, replays and preview | Not run: no `/dev/video*` device visible; Tk unavailable |

Tests cover small ideal rotations, stillness, short windows, moving planar media,
shared translational/projective scene motion, missing texture, jitter, unreliable
pose history breaks, bounded/masked private buffers, native exceptions, normal
historical finish, timeout during inference, loss, quality failure, staleness, cancellation, cleanup failure,
NaN/unknown/vector IPC and unsupported success. Earlier camera/tracking/watchdog
checks remain passing.

The tests exposed artificial background corners created by the face-mask border
on a textureless wall. Feature detection now excludes that border. Subpixel image
warping/downsampling also produces accumulated flow bias: the translation test
retains roughly 13% residual relative to predicted common motion. Its engineering
tolerance checks that most motion is explained, while requiring neutral scores;
it does not demand exact compensation. The production common-motion condition,
score caps and disabled-success policy were not relaxed after this failure.

Final review also found that a new capture timestamp could make an older motion
summary seem fresh when inference completed after timeout. Finish now checks the
last analyzed tracking sample; a regression test requires stale diagnostics and
caches to be cleared in that case.

## Privacy, performance and known failures

All processing is local. The analyzer retains one previous face-masked grayscale
environment image at most 320 × 240 and its mask, plus pose scalars and small private
transforms capped at 64 entries/two seconds. It releases those caches and the
tracker's geometry/appearance on reset or termination. Public logs/JSON contain
only scalar diagnostics, not images, landmarks, pose vectors or embeddings.
Releasing references is not a guarantee of secure RAM erasure.

Motion runs only during the bounded requested capture, at the existing target
12 processed frames per second. Sparse flow and reduced imagery bound work, but
real laptop CPU/latency profiling remains a later milestone.

- A coherent prerecorded video can gain both motion scores; this component cannot
  reliably identify that replay on its own.
- Moving prints/displays can produce changing model landmarks that do not follow
  a perfect plane. Residuals can be model noise rather than real face deformation.
- Pose depends on approximate intrinsics and generic anatomy. Glasses, occlusion,
  lens distortion, low resolution and illumination can make it unavailable or noisy.
- Background parallax, moving objects, rolling shutter, reflections, autofocus
  and interpolation can corrupt compensation. Missing texture/coverage reduces
  evidence; even accepted flow is not a depth measurement.
- An unusually still person, tiny baseline or nonlinear movement can supply no
  useful evidence. This never forces a challenge or an automatic SPOOF result.
- Similar-looking substitutions and coherent replay can evade continuity checks;
  track numbers and motion are not identity recognition.

No PAD false-acceptance/false-rejection or physical sensitivity claim is supported.
Software-only RGB evidence is not equivalent to IR/depth authentication. Next is
milestone 4, a nonmandatory blink sequence detector, only after new direction.

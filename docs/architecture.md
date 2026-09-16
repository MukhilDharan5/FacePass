# Component boundaries

The current pipeline is:

```text
GNOME Terminal (ordinary Fedora desktop user)
  → finite harness / worker supervisor
  → worker: V4L2 camera → local Face Landmarker → quality gates → FaceTracker
      → MotionAnalyzer → head_motion / facial_geometry evidence
  → LivenessSession → scalar diagnostics, always non-success
```

The future pipeline will add the remaining evidence producers and fusion
inside the worker. PAM, GDM, GNOME integration and identity recognition are outside
this repository's implemented scope. The private JSON worker pipe is an internal
test-harness detail, not a stable daemon IPC protocol.

## Current session contract

```python
import time
from facepass.landmarks import MediaPipeLandmarker
from facepass.session import LivenessSession

session = LivenessSession(
    lambda: MediaPipeLandmarker("models/face_landmarker.task"),
    max_duration=5.0,
)
session.start()
try:
    # frame: contiguous uint8 NumPy array, H × W × 3, OpenCV BGR order.
    # timestamp: time.monotonic() seconds captured immediately after frame read.
    result = session.process_frame(frame, time.monotonic())
    diagnostics = result.to_dict()  # Scalars only; safe for local diagnostic logs.
finally:
    session.finish()
```

This example describes the API, not a camera loop. The supplied harness provides
the supervised loop. Direct calls to a native model or V4L2 read can block; a future
daemon must preserve process supervision, cancellation and independent deadlines.

Session states: NEW → RUNNING → FINISHED, FAILED or CANCELLED. A session cannot be
restarted. Frame errors terminate it. Multiple faces terminate it as UNCERTAIN with
`multiple_faces`; no face produces INSUFFICIENT_QUALITY while observation continues.
`current_result()` also enforces timeout even when no new frame is submitted.
Cancellation/finish clears transient geometry, tracking history/appearance and
motion caches, and closes the backend. A clean, fresh, quality-stable finish may
retain historical scalar motion diagnostics; those signals become unavailable
with reliability zero. Cancellation or any failure clears all motion diagnostics
and scores. Historical values are never evidence for another attempt.

`session.observation` holds only the most recent single-face geometry. It is never
serialized by `to_dict()`. Consumers must discard any references they keep. It is
cleared before another frame and on face loss, expiry, error or termination. The
separate `session.history` returns immutable original landmark samples from the
current uninterrupted quality-gated track. It excludes raw images. History access
checks freshness; retaining a consumer reference can extend its lifetime, so
consumers must promptly discard references after use and on every reset.

## Current continuity tracking

SEARCHING → ACQUIRING → STABLE describes a new track. Becoming STABLE requires
three consecutive usable frames spanning at least 150 ms by default. No minimum
motion or blink is required. Full-frame face detection still runs every processed
frame rather than restricting the detector to a previous face crop.

The tracker compares center displacement and logarithmic scale change with elapsed
time-aware limits, and aligns relatively rigid facial anchors by translation,
isotropic scale and roll before comparing shape. An initial shape/appearance anchor
is retained alongside the preceding sample to limit drift. Eyelids, irises and
mouth are excluded from continuity shape anchors, but all raw landmarks remain in
history so later detectors can measure their subtle movement without smoothing.

A small eye/nose-aligned skin region supplies a normalized grayscale appearance
feature when texture is usable. Mean subtraction and unit normalization tolerate
uniform brightness/positive contrast changes; they do not account for every
directional light or reflection change. Missing texture reduces continuity score
instead of forcing rejection. The feature remains private memory, never logging
or IPC. No recognition embeddings or identity database are used.

Loss or poor quality changes the track to LOST and clears all samples, anchor
geometry, appearance features and signal scores. A subsequent quality face starts
ACQUIRING with a new track number. A gap over 350 ms clears history even without a
new frame; `current_result()`, history and observation access enforce freshness.
Short capture gaps may reacquire; the session's existing over-one-second gap rule
remains terminal. Strong geometry/appearance discontinuity or frame resolution
change is REJECTED and terminates the session as UNCERTAIN. Multiple detected faces
remain a terminal ambiguity failure. No missing frame is interpolated into history.

The history ring retains at most 64 samples and at most two seconds by default,
plus a fixed initial geometry anchor. Samples use owned, bytes-backed read-only
arrays so a backend or consumer cannot mutate earlier detector input. Finish or
cancellation closes the track and clears retained data. The public `tracking`
object contains only scalar states, counters and mismatch measurements. Its
`track_id` is an attempt-local sequence number, not a biometric identity.

STABLE means heuristic continuity of observations, not proof of the same person,
3D geometry or liveness. Similar-looking replacements and coherent replay can pass.
Tracking continuity score is an engineering margin and must not be fused as an
independent liveness factor.

The harness bounds startup to 20 seconds, stalls to 2 seconds, and observation to
the requested duration plus a 0.5-second transport/cleanup allowance. Cancellation
requests a graceful worker exit, then kills a blocked worker. A clean, terminal
result is only accepted after the worker exits cleanly. Malformed/unknown/oversized
JSON, a partial packet hang, non-finite values, process failure or an unsupported
decision produces UNCERTAIN. None of these paths can grant authentication.

## Experimental head motion and geometry (milestone 3)

`MotionAnalyzer` receives the immutable track history and current BGR frame. It
performs no capture or model calls. Two pure, replaceable functions in
`motion_evidence.py` produce head-motion and facial-geometry `SignalResult`s.
Neither decides authentication; overall confidence remains exactly zero.

`head_pose.py` uses six image landmarks with a generic engineering 3D template and
OpenCV PnP to estimate yaw, pitch and roll. Intrinsics use a centered principal
point and focal length equal to the larger image dimension; lens distortion and
individual anatomy are uncalibrated. Behind-camera, implausible and high-error
solutions are unavailable. Non-finite native outputs and exceptions fail closed.
Predicted landmark z is not used as measured depth.

A two-second scalar pose history measures percentile angular span and linear-trend
coherence. Six samples spanning 450 ms and STABLE tracking are the minimum evidence
window. An unavailable pose breaks the rotation history. Small coherent rotations
can supply evidence; stillness and irregular/noisy estimates remain neutral. A
natural nonlinear motion can also remain neutral under this conservative coherence
test. No head movement or challenge is mandatory.

Environment flow uses at most 80 corners outside the detected face and a surrounding
margin, a reduced grayscale image at most 320 × 240, and forward/backward optical
flow checks. The artificial black mask border is excluded from feature detection.
RANSAC requires at least 12 inliers, 70% agreement and spatial coverage across the
scene. A similarity transform is preferred unless a projective homography clearly
improves fit. Transform steps compose over the same temporal baseline used for face
landmark displacement. Missing texture, poor coverage or a missing step makes
background-relative evidence unavailable. Shared scene motion suppresses a
positive motion score when its independent residual is small.

A face homography is fitted on 12 spatially distributed anchors and evaluated on
10 different, held-out anchors, avoiding eyelid/iris/mouth animation. Its median
prediction error, normalized by landmark displacement with a noise floor, is a
planar-fit diagnostic. A good planar explanation stays neutral. Residual deformation
accompanied by coherent pose motion and usable environment tracking supplies only
weak positive evidence; model jitter and prerecorded animation can produce it too.

Positive motion scores are capped at 0.68; head reliability is at most 0.4 and
geometry reliability at most 0.35. These are heuristic caps, not calibrated
probabilities or a LIVE threshold. Both producers share landmarks, pose, planar fit
and background flow. Future fusion must group their dependence instead of adding
them as independent security votes. Neither missing motion nor planarity supplies
automatic negative spoof evidence.

The analyzer retains one previous face-masked environment image and mask, plus at
most 64 scalar pose samples and 64 small private scene transforms over two seconds.
No full previous camera frame, recognition embedding or raw vector is serialized.
Track change, loss, quality failure, staleness, rejection, exceptions, cancellation
and termination release the caches. Only a clean finish may retain the explicitly
historical scalar summary described above. IPC validates scalar fields, finite
ranges, experimental score caps and current evidence context; unknown/vector
fields and any LIVE result are rejected.

Algorithm references: [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html),
[optical flow](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html) and
[homography/robust estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

## Detector contract and remaining producers

Each detector receives a quality-gated, continuity-checked observation history
and return `SignalResult(confidence, reliability, available, diagnostic_information)`.
No detector decides authentication. Availability and reliability describe
whether evidence can be trusted in the current capture; absent gaze or no natural
blink is missing evidence, not automatic spoof evidence.

Future stateful detectors must reset their own feature caches on track change,
history reset, quality loss, staleness and termination; retaining scores or cache
state across these boundaries is forbidden. Milestone 3 implements those resets
for motion; later producers must provide their own reset behavior.

Implemented and planned producers (separate modules do not establish statistical independence):

| Producer | Evidence to develop | Main confounders |
| --- | --- | --- |
| Blink | Open → closing → closed → opening → open sequence, bilateral agreement, duration and hysteresis | Tracking noise, glasses, prerecorded natural blinks |
| Head motion, experimental implementation | Approximate pose span/coherence and background/planar confounder controls | Planar media being moved, camera motion, replay motion |
| Facial geometry, experimental implementation | Held-out residual deformation after fitting a planar transform | Model hallucinated geometry, tiny baseline, lens distortion |
| Gaze | Iris position relative to eye corners, independent of head pose | Eye resolution, reflections, glasses, video replay |
| Illumination | Forehead/nose/cheek/chin region shading and color response against outside-face ambient regions across time | Skin reflectance, backlighting, auto-exposure and white balance |
| Replay artifacts | Multi-region frequency/banding/moire/halftone and glare evidence with camera-specific reliability | Sensor noise, fabric, compression, screens with no artifacts |
| Temporal consistency | Agreement between landmarks, pose, gaze, shading, scale and global motion | Correlated features, static scene, sophisticated coherent video |

For illumination, do not equate face/background brightness. Exclude the face from
ambient samples; compare directional face-region shading, RGB balance and temporal
responses after accounting for auto-exposure/white-balance changes. Lighting that
does not change supplies little temporal evidence. A self-illuminated screen can
still respond to room light; lighting consistency alone must never imply LIVE.

Fusion will need empirical calibration and dependence-aware grouping: blink, gaze
and landmarks share an input model and must not count as three fully independent
security factors. Motion/geometry consistency also share measurements. The result
must retain every individual score, evidence availability and quality reasons.
There is intentionally no guessed weighted-sum LIVE threshold in milestone 3.

An adaptive observer will finish early only after sustained, diverse positive
evidence and acceptable quality exceed validated policy. It will extend the window
for moderate evidence, return UNCERTAIN at its hard limit, and return SPOOF only on
credible negative evidence. It must never lower thresholds following a failure.

## Optional challenges (future API design)

Passive mode remains the default. A later challenge controller can have states
PASSIVE_OBSERVING → PASSIVE_UNCERTAIN → CHALLENGE_ISSUED → CHALLENGE_OBSERVING →
COMPLETED/EXPIRED/CANCELLED. Issue a fresh cryptographically random nonce and action
per attempt; bind a bounded response window to the same continuously tracked face.
Reject stale responses, face switches, retries of the same nonce and expiry.
Challenge completion is evidence for fusion, not an authentication bypass. No
challenge issuance or response checking is implemented in this milestone.

## Future daemon boundary

Only after attack evaluation and a stable interface, a separate daemon may combine:

```text
identity_match AND liveness == LIVE AND capture_quality_acceptable AND policy_allows
```

Unknown identities do not enroll themselves. A live unknown person and a recognized
photo both fail authentication. Authentication IPC must add request binding,
freshness, peer credentials, cancellation and fail-closed handling; the harness's
private diagnostic pipe must not be reused as a login interface.

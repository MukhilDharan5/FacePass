# Component boundaries

The current pipeline is:

```text
GNOME Terminal / KDE Konsole / headless terminal (ordinary Fedora user)
  → finite harness / worker supervisor
  → worker: V4L2 camera → bounded dim-scene landmark input → local Face Landmarker → exposure-aware quality gates → FaceTracker
      → MotionAnalyzer → head_motion / facial_geometry evidence
      → BlinkAnalyzer → blink evidence
      → GazeAnalyzer → gaze evidence
      → IlluminationAnalyzer → illumination_consistency evidence
      → ReplayArtifactAnalyzer → replay_artifact_detection evidence
      → TemporalConsistencyAnalyzer → temporal_consistency evidence
      → dependence-aware group summary → FusionDiagnostics
      → AdaptiveObservationPolicy → ObservationDiagnostics
  → LivenessSession → scalar diagnostics, always non-success
```

The temporal and fusion diagnostics are implemented inside the worker, but are not
an authentication policy. Version 0.14.0 adds a separate experimental KDE/PAM
authenticator, described in [face-unlock.md](face-unlock.md). The research results
never authorize it. The private JSON worker pipe is an internal
test-harness detail. The separate versioned research daemon wraps this supervisor;
its public local contract is documented in [daemon-protocol.md](daemon-protocol.md).

Version 0.13.0 applies a bounded full-frame gamma lookup only to dim landmark
input in the default tolerant profile. Quality gates and all image-based
detectors receive the original frame. Brightness-adjusted sharpness and contrast
allow moderate blur/dim detail to remain on the current track with warnings;
hard quality failures still clear history. Strict capture keeps the original
gates and skips enhancement. See [capture-quality.md](capture-quality.md).

Desktop detection is a read-only diagnostic in `desktop.py`, using
`XDG_CURRENT_DESKTOP` and session/display environment hints. It does not gate
V4L2 capture, model processing, evaluation or the Unix-socket daemon. The optional
Tk window connects to X11, including XWayland under GNOME and KDE Plasma Wayland.
No Qt platform plugins, compositor API or desktop extension is required. Doctor
reports configured prerequisites without opening a display or camera; preview
initialization failures terminate before spawning a capture worker. See
[Fedora KDE setup](fedora-kde.md).

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
the supervised loop. Direct calls to a native model or V4L2 read can block; the
research daemon preserves process supervision, cancellation and independent deadlines.

Session states: NEW → RUNNING → FINISHED, FAILED or CANCELLED. A session cannot be
restarted. Frame errors terminate it. Multiple faces terminate it as UNCERTAIN with
`multiple_faces`; no face produces INSUFFICIENT_QUALITY while observation continues.
`current_result()` also enforces timeout even when no new frame is submitted.
Cancellation/finish clears transient geometry, tracking history/appearance and
  motion/blink/gaze/illumination/replay/temporal caches, and closes the backend. A clean, fresh, quality-stable finish may
retain historical scalar detector diagnostics; those signals become unavailable
with reliability zero. Cancellation or any failure clears all detector diagnostics
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

## Experimental bilateral blink sequences (milestone 4)

`BlinkAnalyzer` receives the same immutable history and STABLE flag, independently
of the motion analyzer. It uses only the current sample and its own constant-size
temporal state. It performs no camera/model calls or identity recognition.

Six points per eye provide pixel-space eye aspect ratio (EAR): the sum of two
upper/lower eyelid distances divided by twice the eye-corner distance. Multiplying
normalized x/y by image width/height avoids an aspect-ratio error. Rigid in-plane
roll and uniform scale preserve the ratio; perspective and anatomy still affect it.

The initial engineering gates are:

| Gate | Value / handling |
| --- | --- |
| Both eyes open | Each EAR ≥ 0.23 |
| Both eyes closed | Each EAR ≤ 0.15; intervening values supply hysteresis |
| Eye geometry | Width ≥ 18 px; in-frame, plausible and non-crossed eyelids |
| Bilateral agreement | EAR difference ≤ 0.10 and width ratio ≤ 1.8 |
| Width continuity | Per-eye absolute logarithmic width change ≤ 0.2 per sample |
| Open/closed confirmation | At least two consecutive qualifying samples spanning ≥ 60 ms |
| Initial signal observation | At least three usable eye samples spanning ≥ 150 ms |
| Complete sequence | Confirmed reopening within 650 ms of the first below-open sample |
| Sampling freshness | Gap ≤ 220 ms; larger gaps clear blink state |
| Positive evidence lifetime | At most two seconds after confirmed completion, while phase is OPEN |

```text
ACQUIRING_OPEN → OPEN → CLOSING → CLOSED → OPENING → OPEN
                                  └─ prolonged closure → SUSTAINED_CLOSED
```

The full sequence supplies weak evidence. A transition directly from open to a
closed sample is allowed because intermediate motion may fall between camera
frames, but closure and reopening still require separate consecutive confirmations.
Initially closed eyes do not establish the open baseline. An isolated collapsed
sample, separated spikes, an interrupted reopening or an overlong sequence does
not count. Held-closed eyes must regain a confirmed open baseline before a new
cycle. Poor geometry or asymmetry clears the blink detector, making this signal
unavailable without rejecting an otherwise usable face. Missing/no blink is neutral,
never negative spoof evidence or a request for a deliberate action.

Completed blink confidence is capped at 0.65 and reliability at 0.35. Valid but
unrewarded observations are neutral (0.5, reliability 0.15); unreliable/short
observations are unavailable with reliability zero. These scores are not measured
probabilities. Blink and motion share MediaPipe landmarks and must be grouped as
dependent evidence in future fusion. A prerecorded video can contain the same blink.

`blink` JSON contains current scalar EAR/width, phase, usable observation count/span,
completed/rejected sequence counts, timing and an event code. Blink duration is
the first below-open sample to the first sample of the confirmed reopening run;
confirmation adds latency and does not establish exact physical eyelid timing.
`closed_duration_ms` is an elapsed timer from the first consecutive closed sample
in the current cycle, cleared when the cycle ends. No raw eye trajectories or
landmark vectors are serialized. The detector stores only current scalar diagnostics,
two previous widths, counters and phase timestamps.

All continuity/quality/reset/terminal paths release state. `current_result()` polls
the 220-ms cadence limit and two-second credit expiry even without a new frame;
tracking may remain STABLE between 220 and 350 ms while blink is unavailable.
Strict worker IPC checks the scalar schema, finite bounds, evidence context,
sequence duration/freshness and experimental caps. A clean finish may retain
historical scalars with availability false/reliability zero; failures clear them.

At 12 processed fps, a fast physical blink may lack two closed observations.
Glasses, occlusion, individual anatomy, perspective and model noise are unmeasured
confounders; geometry checks are not calibrated visibility estimation. There is no
physical blink sensitivity or replay-resistance result yet.

Algorithm references: [Soukupová and Čech, Real-Time Eye Blink Detection using Facial
Landmarks](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-CVWW-2016.pdf) for the
EAR measurement, and [MediaPipe eye contour connections](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py)
for landmark indices. This implementation uses an engineering state machine, not
the paper's trained temporal classifier or its validated results.

## Experimental relative iris movement (milestone 5)

`GazeAnalyzer` independently receives the current sample from the stable immutable
track. `gaze_evidence.py` holds a replaceable weak evidence function. Four ordered
iris-rim points per eye provide a centroid. Its horizontal/perpendicular displacement
from the image-left eye corner is divided by eye width. Pixel-space coordinates
remove translation, uniform scale and in-plane roll, but not perspective effects.
This proxy does not estimate gaze angles, screen position, pupil physiology or identity.

Each eye must be open (EAR ≥ 0.23), at least 28 px wide, with both iris diagonals
≥ 6 px and a plausible convex ring. Diameter/width must be 0.15–0.5; opposite
diameters, centers and directions must agree. The centroid must be in the eye
aperture, with bounded rim coordinates. Rim points may be hidden behind lids;
these are geometry checks, not measured visibility. Eye asymmetry, a geometry log
step over 0.2 or an iris-position step over 0.12 eye widths clears gaze state.
One unusable eye makes gaze unavailable while a quality face can remain STABLE.

The same generic approximate pose estimator supplies yaw/pitch/roll. Missing pose,
|yaw| > 25° or |pitch| > 20° clears the gaze window. Non-finite/native errors fail
the session closed. At most 64 scalar samples spanning two seconds hold timestamp,
two x/y pairs and three pose angles; no additional image or full landmark cache.
All temporal reset and termination boundaries release this history.

Six usable samples spanning 450 ms are required. Robust displacement compares the
first and last three-sample medians in each eye; both must move at least 0.035 eye
widths and two pixels relative to the current smaller eye width. Joint linear-fit
coherence and bilateral direction/magnitude agreement must each be ≥ 0.70. Agreement
combines a nonnegative direction cosine with the smaller/larger displacement ratio.
Percentile spans are diagnostics, not sufficient evidence by themselves. A single
spike, fixed other eye, opposed motion or unstructured jitter does not earn credit
in generated tests. Conservative trend gating can miss genuine nonlinear movement.

Head rotation span > 1.5° suppresses positive gaze evidence and reduces neutral
reliability to 0.05; otherwise neutral reliability is 0.10. This gate conservatively
avoids rewarding coupled head movement, without claiming calibrated 3D compensation.
Independent iris motion is capped at confidence 0.62/reliability 0.30. No movement
stays neutral and no deliberate gaze direction is requested. Replayed eye movement
can satisfy all these conditions; blink, gaze and motion share model dependence.

The 220-ms cadence limit and two-second window are also enforced by `current_result()`.
Old samples expire during stillness and between new frames; no score or sample joins
across gaps. JSON exposes only current scalar positions, resolution, window counts,
span/coherence/agreement, head span, suppression and event flags. Strict IPC validates
finite types/bounds, observation context, score caps and independent-motion conditions.
Clean finish marks scalar summaries historical and evidence unavailable; failure,
cancellation or poor/stale capture clears them.

References: [MediaPipe iris documentation](https://chuoling.github.io/mediapipe/solutions/iris.html)
describes iris landmarks and distinguishes them from the location a person looks
at; [official contour connections](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py)
define the iris rims. This engineering temporal proxy has no physical accuracy or
presentation-attack evaluation result. Glasses/occlusion are not directly classified.

## Experimental multi-region illumination (milestone 6)

`IlluminationAnalyzer` receives the current BGR frame, immutable stable track and
the already-computed `MotionDiagnostics`. It performs no camera/model calls and
does not read or change camera settings. `illumination_evidence.py` is a separate,
replaceable evidence function; it never decides authentication.

Landmarks locate forehead, nose, both cheeks, both subocular socket patches and
chin. Temporary circular patches lie inside an eroded face hull and cannot overlap
substantially. Four separated quadrant tiles supply environment samples outside
a dilated hull, excluding the face and its immediate boundary. Masks and a color
frame are temporary and at most 320 × 240. Face patches need at least 24 pixels;
each ambient tile needs 128. More than 10% clipped pixels in any region or a very
dark median channel makes this producer unavailable. The clipped fraction uses
all region pixels; medians linearize at most 512 regularly sampled pixels per region.

BGR values are converted to RGB and approximately inverse-sRGB linearized. Median
RGB per patch gives ambient luminance/balance and static regional log contrasts
(cheeks, forehead/chin, nose/cheeks and sockets). The webcam's transfer function,
tone mapping, reflectance and spectral response are uncalibrated. RGB balance
describes observed scene colors, not calibrated illuminant color temperature.
Static contrasts are diagnostics, never a face/background mean-brightness test.

For every sample, log RGB changes relative to the first sample are calculated.
The median change across the four ambient tiles, separately per channel, is
subtracted from both face and ambient changes. This removes a common multiplicative
gain/color shift as a possible exposure/WB effect. It also removes actual uniform
room-light changes; those cannot be disentangled from camera gain in these images.
`global_gain_ev` is this observed common gain in stop units, not camera exposure
metadata; `common_color_shift` is the spread of its channel log gains.

Fixed x/y/intercept least-squares fits describe spatial RGB response over seven
anatomical face coordinates and four quadrant coordinates. First/last three-sample
medians give gradient change, its nonnegative directional cosine and the number
of responding face regions. Centered gradient trajectories provide temporal
coherence. RGB gradients are compared after the best nonnegative scalar alignment.
A face-region RMS residual after the spatial fit requires a response richer than
one simple x/y gradient. These coordinates are image-plane proxies, not surface
normals or measured depth; residual is neither physical planarity nor print proof.
Reported angles describe response direction in those coordinates, not the true
direction of a room light. No calibrated Kelvin or emitter direction is inferred.

| Initial engineering gate | Value |
| --- | --- |
| Pose | Available, absolute pitch ≤ 20°, yaw ≤ 25° |
| Evidence window | ≥ eight samples spanning ≥ 600 ms |
| History/freshness | ≤ 64 samples over ≤ two seconds; gaps over 350 ms reset |
| Abrupt common radiometric step | Any channel's median ambient log step > 0.35 resets |
| Spatial ambient / face gradient change | ≥ 0.05 / ≥ 0.03 in log-response units |
| Responding face regions | ≥ four with RGB RMS change ≥ 0.03 |
| Regional response residual | ≥ 0.012 |
| Temporal coherence / direction agreement | Each ≥ 0.80 |
| RGB gradient alignment error | ≤ 0.35 |
| Motion context for positive evidence | Background flow available for the entire window |
| Confounder limits | Pose span ≤ 1.5°, face-center drift ≤ 2 px, common scene motion ≤ 2 px |

Scene motion comes from milestone 3's composed background transform and may cover
a longer baseline than the current illumination window, making suppression
conservative. Missing flow context or excessive motion stays neutral at reliability
0.03. Qualifying response is capped at confidence 0.58/reliability 0.20; ordinary
valid observations are neutral at 0.50/0.08. A substantial ambient change with
absent, opposed, incoherent or spectrally different facial response sets an
unvalidated mismatch flag and stays neutral at 0.50/0.05. Short/unreliable
observations are unavailable with zero reliability. No light change is required.

Private memory holds timestamp, seven facial/ four ambient log-RGB summaries,
pose, face center and background context per sample. No image, mask or raw pixel
trajectory is retained or serialized. Track changes, loss, quality/stale capture,
native errors, cancellation and terminal failures release all state. Polling trims
history and checks freshness. A clean finish can retain historical scalar fields,
with availability false and reliability zero, after private buffers are released.
`_refresh_temporal_signals` refreshes blink, gaze and illumination between frames;
exceptions during polling or finish fail as `temporal_evidence_error` and clear all
detector summaries. Strict worker IPC checks the known scalar schema, finite types
and bounds, experimental caps, stable quality context and every positive-response
gate; it rejects vectors, impossible contexts and LIVE.

Reflectance, beards/glasses, mixed light, moving objects, specularities, occlusion,
clipping and camera local processing can defeat these assumptions. Hue-only
changes or nonlinear genuine response can remain neutral. A room-lit print can
respond, and a display/replay with matching response can earn the same weak score.
All producers share landmark and scene/pose dependencies; fusion must account for
them. There is no measured physical lighting accuracy or PAD performance.

References: [W3C sRGB conversion code](https://www.w3.org/TR/css-color-4/#color-conversion-code)
defines the transfer used as an approximation. Linux V4L2 documents
[automatic exposure and white-balance controls](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html);
their existence motivates common-gain removal, but this implementation does not
claim to identify the actual control state from frames.

## Experimental replay/print artifact analysis (milestone 7)

`ReplayArtifactAnalyzer` receives the current BGR frame, stable immutable track and
milestone 3 motion diagnostics. It performs no capture/model calls. Five equal-size
patches centered on forehead, nose, both cheeks and chin must lie inside an eroded
face hull. Four equal-size corner patches must lie outside a dilated hull. At up to
640 × 480 working resolution, each patch is resampled temporarily to 32 × 32; the
analyzer keeps feature scalars only and immediately releases patches and transforms.

Each patch supplies seven bounded, uncalibrated diagnostics:

| Family | Measurement | Major confounders |
| --- | --- | --- |
| Periodic/moire/halftone | Concentration of windowed 2D luminance Fourier energy between 0.08 and 0.45 cycles/pixel | Fabrics, hair, regular room texture, demosaicing |
| Display chroma grid | Periodic concentration in max RGB opponent differences, scaled by chroma/luminance contrast | Color fabric, sensor chroma aliasing, compression |
| Refresh/banding | Strongest non-DC row/column-mean spectrum | Rolling shutter, LED room lights, blinds |
| Sharpening | Laplacian magnitude relative to gradient magnitude | Camera sharpening, facial hair, noise |
| Resampling | Maximum absolute luminance autocorrelation at horizontal/vertical lags 2–6 | Natural repeating texture, denoising, interpolation |
| Glare/reflection | Near-white low-chroma pixel fraction | Oily skin, glasses, direct lights, overexposure |
| Flat/axis edge | Fraction of strong gradients close to horizontal or vertical | Architecture, hairlines and grid texture |

The face median for each family is reduced by 75% of the background median. This
is only a camera/scene control; it does not locate a phone/print boundary or prove
that face and background should share frequency content. The maximum raw background
family score must be ≤ 0.45. At least three face regions must cross a family gate.
Weak suspicion requires at least two families present in 75% of frames over six
samples/450 ms. A transient artifact, one glare patch, background-only pattern or
camera-wide banding stays neutral in controlled tests. Low texture makes this
producer unavailable. No single artifact is required or sufficient.

The analyzer reuses background-flow availability and common motion from milestone
3. Face-center drift over 3 px or common scene motion over 3 px suppresses suspicious
evidence because changing patch contents and blur can manufacture spectral changes.
Unavailable flow also reduces reliability. This can miss an attack or suppress a
real artifact; it is a conservative confounder gate rather than motion compensation.

| Observation | Confidence / reliability | Availability |
| --- | --- | --- |
| Short, small or low-texture regions | 0.50 / 0 | False |
| Usable regions, no persistent multifamily artifact | 0.50 / 0.10 | True |
| Missing/moving scene context | 0.50 / 0.03 | True after the minimum window |
| Persistent multifamily regional artifact | 0.35 / 0.25 | True |

Confidence is expressed on the same liveness-oriented scale as other producers,
so 0.35 is weak suspicion. It is not an attack probability and does not return
SPOOF. Absence of an artifact remains 0.50 neutral, never positive live evidence.
Candidate observations cannot change overall confidence from zero because a
validated authentication fusion policy is not implemented.

Private state contains timestamp, 5 × 7 face and 4 × 7 background scores, patch
width/contrast, face center and motion context for at most 64 samples/two seconds.
It retains no image, patch, FFT, autocorrelation, mask or device boundary. A 350-ms
gap, track reset/loss, poor quality, stale capture, exception, failure or cancellation
clears it. Polling expires stale samples. A clean finish can retain historical
scalar diagnostics with evidence unavailable and reliability zero. Strict worker
IPC admits only known finite scalar fields and enforces score, region, persistence,
background and motion conditions; LIVE and SPOOF remain unsupported.

The generated patterns merely exercise the calculations. A generated pattern
labeled as bona fide gets the same suspicious result as the display fixture,
demonstrating that pixel structure cannot reveal its origin. Real camera pipelines
can erase or manufacture all seven signals. Physical evaluation across cameras,
distances, displays, prints, skin, hair, glasses, fabrics, rooms and compression is
required before thresholds can support a PAD policy.

## Detector contract and remaining producers

Each detector receives a quality-gated, continuity-checked observation history
and return `SignalResult(confidence, reliability, available, diagnostic_information)`.
No detector decides authentication. Availability and reliability describe
whether evidence can be trusted in the current capture; absent gaze or no natural
blink, illumination change or replay artifact is missing evidence, not automatic spoof evidence.

Future stateful detectors must reset their own feature caches on track change,
history reset, quality loss, staleness and termination; retaining scores or cache
state across these boundaries is forbidden. Milestones 3–7 implement those resets
for motion, blink, gaze, illumination, replay artifacts and temporal consistency; later producers must provide their own reset behavior.

Implemented and planned producers (separate modules do not establish statistical independence):

| Producer | Evidence to develop | Main confounders |
| --- | --- | --- |
| Blink, experimental implementation | Open → closing → closed → opening → open sequence, bilateral agreement, duration and hysteresis | Tracking noise, glasses, prerecorded natural blinks |
| Head motion, experimental implementation | Approximate pose span/coherence and background/planar confounder controls | Planar media being moved, camera motion, replay motion |
| Facial geometry, experimental implementation | Held-out residual deformation after fitting a planar transform | Model hallucinated geometry, tiny baseline, lens distortion |
| Gaze, experimental implementation | Relative iris motion, bilateral/coherence checks and conservative head-rotation gate | Eye resolution, reflections, glasses, head coupling, video replay |
| Illumination, experimental implementation | Seven facial patches vs four outside-face tiles, temporal spatial/RGB response and common-gain removal | Reflectance, local tone mapping, mixed light, pose/scene motion and coherent replay |
| Replay artifacts, experimental implementation | Multi-region frequency/banding/moire/halftone, chroma, sharpening, resampling, glare and edge diagnostics | Sensor processing, fabric/room patterns, compression, high-quality media with no artifacts |
| Temporal consistency | Agreement between landmarks, pose, gaze, shading, scale and global motion | Correlated features, static scene, sophisticated coherent video |

For illumination, do not equate face/background brightness. Exclude the face from
ambient samples; compare directional face-region shading, RGB balance and temporal
responses after accounting for auto-exposure/white-balance changes. Lighting that
does not change supplies little temporal evidence. A self-illuminated screen can
still respond to room light; lighting consistency alone must never imply LIVE.

## Temporal consistency and dependence-aware fusion (milestone 8)

`TemporalConsistencyAnalyzer` receives only the current scalar detector results and
tracking/motion context. It stores at most 64 rows over two seconds and rejects
duplicate/non-finite timestamps. A gap over 350 ms, unstable tracking, quality loss,
staleness, failure or termination clears the rows. Polling refreshes the latest row
after blink/gaze/illumination/replay expiry without extending its timestamp.

The group map is explicit: `landmark_dynamics` contains head motion, facial geometry,
blink and gaze; `photometric_scene` contains illumination response; and
`replay_texture` contains replay suspicion. Landmark support uses the strongest
normalized member rather than summing correlated signals. Replay suspicion can create
a conflict, never positive live support. The analyzer reports support overlap,
cross-group agreement, temporal stability, context suppression and missing groups.

`FusionDiagnostics` keeps group scores, coverage, raw support, an explicit dependence
penalty and adjusted support for review. These values are engineering diagnostics,
not probabilities or a guessed weighted-sum threshold. A two-group window may expose
weak temporal support capped at 0.56/0.18, but `liveness_candidate` and
`spoof_candidate` remain false and `LivenessResult.confidence` remains zero.

Fusion will need empirical calibration and dependence-aware grouping: blink, gaze
and landmarks share an input model and must not count as three fully independent
security factors. Motion/geometry consistency also share measurements. The result
must retain every individual score, evidence availability and quality reasons.
There is intentionally no guessed weighted-sum LIVE threshold in milestone 8.

## Adaptive observation policy (milestone 9)

`AdaptiveObservationPolicy` controls capture duration using only scalar result
diagnostics. The default minimum window is 1.5 seconds. At least eight samples over
600 ms of stable, quality-acceptable two-group support can mark a diagnostic window
ready for early completion when it has no conflict, motion suppression or replay
candidate. This state ends capture only; global confidence remains zero.

At the base duration, partial conflict-free evidence may request one two-second
extension. The effective limit is bounded and cannot be extended twice. Conflicted,
unstable or poor-quality evidence receives no extension and reaches the hard timeout.
`adaptive=False` disables both extension and early completion. Challenge issuance is
explicitly disabled and represented as a scalar `challenge_state="disabled"`.

Policy counters and limits are serialized in `ObservationDiagnostics`. Session reset
and release paths clear them with detector evidence; only a fresh, stable clean finish
may retain historical policy scalars. The policy is never an authentication endpoint,
and no timeout or early completion can emit LIVE or SPOOF.

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

## Local research service and future authentication boundary

The implemented research daemon uses a versioned Unix socket with same-user peer
credentials, one supervised finite worker, cancellation and bounded observation.
It exposes only diagnostics and keeps authentication disabled. See
[daemon-protocol.md](daemon-protocol.md). Evaluation records and profiles are
separate optional tooling; their outputs cannot change a runtime decision.

Only after attack evaluation and a validated policy, an authentication component may combine:

```text
identity_match AND liveness == LIVE AND capture_quality_acceptable AND policy_allows
```

Unknown identities do not enroll themselves. A live unknown person and a recognized
photo both fail authentication. Authentication IPC must add request binding,
freshness, peer credentials, cancellation and fail-closed handling; the harness's
private diagnostic pipe must not be reused as a login interface.

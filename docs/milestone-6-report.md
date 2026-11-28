# Milestone 6 — experimental illumination response

Version 0.6.0 adds optional multi-region facial and outside-face ambient photometry
to the standalone Fedora GNOME harness. It compares spatial shading and RGB
response over time after removing common channel gains as possible exposure or
white-balance effects. It does not establish live skin, identity, calibrated light
direction/temperature or facial depth. Overall confidence remains zero, LIVE is
disabled and every capture run exits 2. No PAM, GDM, GNOME, sudo or system
authentication configuration was changed. Development pauses before milestone 7.

## Implementation and assumptions

`illumination.py` contains `IlluminationAnalyzer`; `illumination_evidence.py` holds
a replaceable evidence function. Both run locally. Each quality-accepted STABLE
track supplies current pixels and landmarks; milestone 3 supplies approximate
pose and background-motion diagnostics without another model inference.

Seven landmark patches cover forehead, nose, both cheeks, both subocular socket
areas and chin. Temporary circles must have sufficient support inside an eroded
face hull and cannot substantially overlap. Four separated quadrant tiles sample
the environment outside a dilated face hull. This excludes detected facial pixels
and a boundary margin; it cannot classify every hair, occlusion or moving object.
Geometry gates do not guarantee that a patch contains visible skin.

Temporary masks and a color frame are at most 320 × 240. Clipping uses all pixels
in each region, while RGB medians linearize at most 512 regularly sampled pixels
per region. Approximate inverse-sRGB conversion follows the
[W3C transfer formula](https://www.w3.org/TR/css-color-4/#color-conversion-code);
the webcam transfer, local tone mapping and spectral response are uncalibrated.
Regional medians provide current ambient luminance and normalized RGB balance,
plus cheek, forehead/chin, nose/cheek and socket log-luminance contrasts. Static
shading and mean face/background brightness differences never earn positive credit.

The bounded history stores log-RGB summaries. For each channel, the median ambient
change across tiles relative to the first sample is subtracted from both face and
ambient changes. This accounts for a common multiplicative exposure/WB shift,
without claiming to identify the camera's actual settings. True uniform room-light
changes are also removed and intentionally stay neutral. Linux V4L2 documents
[automatic exposure and white-balance controls](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html);
this producer does not read or alter them.

Fixed image-plane x/y/intercept fits characterize spatial RGB response in the face
and environment. First/last three-sample medians provide gradient changes, direction
agreement and responding-region counts. Centered gradient histories give temporal
coherence; best nonnegative scalar alignment gives RGB gradient error. A face-region
residual requires response richer than a simple spatial gradient. This is an
engineering diversity gate, not measured surface normals, depth, physical
planarity or a validated print classifier.

Ambient RGB balance describes observed colors, not calibrated illuminant
chromaticity or Kelvin temperature. Direction angles are spatial-response proxies,
not the actual direction of an emitter. `global_gain_ev` reports the removed common
gain in stop units, not camera exposure metadata. Mixed lighting, wall reflectance,
skin/facial hair, glasses, highlights, shadows and camera processing all affect
these quantities. Physical highlights/shadows are not segmented or classified.

| Initial engineering gate | Value |
| --- | --- |
| Region support | Seven face patches ≥ 24 pixels each; four ambient tiles ≥ 128 each |
| Patch clipping/darkness | Clipped pixels ≤ 10%; median linear RGB channels ≥ 0.005 |
| Pose | Available, absolute pitch ≤ 20°, yaw ≤ 25° |
| Evidence window | ≥ eight samples spanning ≥ 600 ms |
| History/freshness | ≤ 64 samples over ≤ two seconds; sample gap ≤ 350 ms |
| Abrupt common gain/color step | Median ambient log step > 0.35 in any channel resets |
| Spatial ambient / face change | ≥ 0.05 / ≥ 0.03 log-response gradient magnitude |
| Responding face regions | ≥ four with RGB RMS change ≥ 0.03 |
| Regional response residual | ≥ 0.012 |
| Coherence / direction agreement | Each ≥ 0.80 |
| RGB alignment error | ≤ 0.35 |
| Motion context | Background flow available for the whole window |
| Motion suppression | Any pose-component span > 1.5°, face-center drift > 2 px or common scene motion > 2 px |

The scene transform can span a longer baseline than the illumination window,
making suppression conservative. Genuine changes can remain neutral, especially
small, hue-only or nonlinear responses. None of these gates is calibrated against
physical bona fide/attack recordings.

| Observation | Confidence / reliability | Availability |
| --- | --- | --- |
| Short, missing or unreliable region/pose observation | 0.50 / 0 | False |
| Usable static/unrewarded illumination | 0.50 / 0.08 | True after minimum window |
| Missing scene-motion context or motion confounder | 0.50 / 0.03 | True after minimum window |
| Ambient/facial response mismatch, otherwise usable | 0.50 / 0.05 | True |
| Qualifying diverse spatial/RGB response | 0.58 / 0.20 | True |

Mismatch means substantial ambient response with absent, opposed, incoherent or
spectrally inconsistent facial response. It is an unvalidated flag, never an
automatic negative score or SPOOF decision. No deliberate light change or other
challenge is required; passive static captures can remain neutral.

Private history contains seven facial and four ambient log-RGB summaries per
sample, timestamp, pose, face center and scene context. No additional image or mask
is retained. Polling trims the window and enforces freshness; old stimulus expires
during stillness. Track changes, loss, quality/stale capture, exceptions, cancellation
and terminal failure clear private summaries and scores. A clean fresh finish may
retain only historical scalar diagnostics, with availability false and reliability
zero after buffers are released. Cleanup failure clears that summary too.

Terminal/preview output shows illumination confidence, availability and response
status. JSON exposes current ambient balance/shading, counts, common gain/color
shift, spatial response/coherence/residual/color error, direction proxies and motion
flags. Strict worker IPC checks the exact scalar schema, finite types/bounds, score
caps, stable quality context and positive-response gates. Region vectors, pixels,
landmarks and embeddings are not serialized. Polling/finish native errors terminate
as `temporal_evidence_error`, release all producers and cannot retain evidence.

## How to test on Fedora GNOME

From the project directory, with the existing venv and verified model:

```bash
.venv/bin/python tools/smoke_illumination.py
.venv/bin/python tools/smoke_illumination.py --json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Select a capture node reported by doctor; run as the ordinary Fedora desktop user.
For mesh preview, omit `--json` and add `--preview` when Tk and a GNOME display are
available. Naturally look at the laptop and inspect region availability, ambient
balance, spatial response, motion suppression and final non-success. Static light
is acceptable. Raw frame retention remains disabled by default. See
[evaluation.md](evaluation.md) for physical cases across users, rooms and media.

The offline tool opens no camera and generates fifteen controlled pixel-gain cases:

| Generated scenario | Confidence / reliability | Expected |
| --- | --- | --- |
| Static side shading / different face-background colors | 0.50 / 0.08 | Neutral |
| Uniform exposure change | 0.50 / 0.08 | Common gain removed, neutral |
| Common white-balance change | 0.50 / 0.08 | Common color gain removed, neutral |
| Diverse spatial response | 0.58 / 0.20 | Weak evidence |
| Spatial response plus common exposure/WB gains | 0.58 / 0.20 | Weak evidence |
| Background-only change | 0.50 / 0.05 | Unvalidated mismatch, neutral |
| Opposed face/background response | 0.50 / 0.05 | Unvalidated mismatch, neutral |
| Simple planar regional gradient | 0.50 / 0.08 | Insufficient response diversity, neutral |
| Spectrally inconsistent response | 0.50 / 0.05 | Unvalidated mismatch, neutral |
| Tiny lighting change | 0.50 / 0.08 | Neutral |
| Missing scene-motion context | 0.50 / 0.03 | Neutral |
| Head rotation | 0.50 / 0.03 | Motion-confounded, neutral |
| Camera movement | 0.50 / 0.03 | Motion-confounded, neutral |
| Sequence crossing a sampling gap | 0.50 / 0 | Reset; final window too short |
| Identical replayed response pattern | 0.58 / 0.20 | Same weak score, origin undetermined |

These fixtures author regional RGB gains to exercise measurement logic. They are
not physical skin, 3D face lighting or display renders and have no ground-truth
liveness label. Standalone cases provide known synthetic pose/scene context; the
complete generated session runs real background-flow math. It finishes UNCERTAIN,
confidence 0, illumination historical/unavailable, and retained tracking/private
history zero. The tool's exit 0 means engineering verification, not authentication.

## Verification performed

Environment: Fedora 44 Workstation, GNOME Wayland, x86_64, Python 3.14.7. Editable
package metadata is 0.6.0. Existing MediaPipe, NumPy, OpenCV and verified model were
reused; no new runtime dependency or system package was required.

| Check | Result |
| --- | --- |
| Full automated suite | 164 tests passed, including 29 new illumination/ROI/session tests |
| Generated illumination demo / JSON | Fifteen cases passed; every complete-session result passed strict IPC validation |
| Real model initialization, blank inference and cleanup | Passed; zero faces, no webcam opened |
| Dependency consistency | `pip check`: no broken requirements |
| Doctor | Milestone 6, authentication disabled, no visible camera, Tk unavailable |
| Missing-camera worker capture | UNCERTAIN, confidence 0, empty history/unavailable illumination, exit 2 |
| Physical lighting, skin, glasses, print/display/replay and GNOME preview | Not run: no camera visible; Tk unavailable |

New tests cover RGB conversion/channel order, independent region-mask support and
outside-face exclusion, clipping/darkness, common exposure/WB removal, spatial
response with common gain, static/tiny/short observations, planar/noisy/spectral
mismatch, missing motion context, head/scene movement, missing/nonfrontal pose,
radiometric jumps, gap/track/quality/stale/terminal resets, bounded histories and
expiry during updates/polls. They also exercise malformed frames/landmarks/clocks,
native update/poll/finish errors, historical cleanup, backend cleanup failure,
post-inference timeout, scalar JSON at 12 and 30 fps, strict IPC and unsupported LIVE.
Existing capture/tracking/motion/blink/gaze/watchdog tests continue to pass.

## Known failures and future work

- A print can respond to room light, while a replay or adaptive/generated display
  can match regional response. The identical-pattern fixture demonstrates the
  limitation; it does not measure physical attack false acceptance.
- Fixed anatomy patches and predicted landmarks do not guarantee visible skin.
  Facial hair, glasses, occlusion, mixed lights and highlights/shadows can corrupt
  otherwise plausible measurements. Forehead/chin patch placement needs physical
  testing across anatomy and pose.
- Ambient colors depend on scene reflectance; response direction is not emitter
  direction. Approximate sRGB, regular subsampling and common-gain removal can fail
  under camera local tone mapping, compression or spatially varying WB/exposure.
  No camera control metadata or calibrated illuminant reconstruction is available.
- Genuine head/camera movement is conservatively suppressed. Close framing, moving
  objects, weak background texture, low cadence or clipped/dark tiles reduce
  availability. Passive static scenes need other evidence, not forced light changes.
- Illumination shares landmarks, pose and scene tracking with the other producers.
  Future fusion must account for dependence and use held-out physical data before
  enabling any LIVE policy. No physical sensitivity, attack accuracy or IR/depth
  equivalence has been demonstrated.

Milestone 7 remains planned. Physical camera/lighting/media evaluation and later
dependence-aware fusion are required; no authentication integration is enabled.

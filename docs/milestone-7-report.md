# Milestone 7 — experimental replay and print artifact evidence

Version 0.7.0 adds conservative multi-region texture/frequency diagnostics for
common display and print artifacts to the Fedora GNOME harness. It does not identify
a physical replay medium or provide validated PAD classification. Overall confidence
remains zero; LIVE and SPOOF are disabled and every camera attempt exits 2. No PAM,
GDM, GNOME, sudo or system authentication files were changed. Development pauses
before milestone 8 temporal consistency and fusion.

## Implementation and assumptions

`replay.py` contains `ReplayArtifactAnalyzer`; `replay_evidence.py` contains the
replaceable weak evidence mapping. Each quality-accepted STABLE track supplies the
current BGR frame and landmarks. Milestone 3 supplies background-flow context; no
extra model inference, camera control, file/network access or challenge is used.

Five square patches center on forehead, nose, both cheeks and chin. Each must have
at least 90% support inside an eroded face hull. Four equal-size controls lie near
the frame corners and need 90% support outside a dilated hull. This samples texture
away from the detected face, but does not locate the boundary of a phone, tablet,
paper or screen. Hair, clothes, room objects and a large presentation medium can
contaminate either group.

The working frame is capped at 640 × 480. Patch width is 22% of inter-eye distance,
with a 17-pixel minimum, and each temporary patch is resized to 32 × 32. At least
three facial patches require luminance standard deviation ≥ 0.012. Low texture,
small/unsupported regions and malformed geometry make the producer unavailable.
Resizing itself changes the spectrum, so face scores are compared with equal-size
outside-face controls rather than interpreted as calibrated physical measurements.

| Family | Implemented diagnostic | Initial family gate | Important confounders |
| --- | --- | --- | --- |
| Periodic / moiré / halftone | Top-four concentration of Hann-windowed 2D luminance FFT energy at 0.08–0.45 cycles/pixel | 0.45 | Fabrics, hair, demosaicing, regular texture |
| Display chroma grid | Same concentration on maximum pairwise RGB difference, scaled by chroma/luminance contrast | 0.48 | Color fabric, sensor chroma aliasing, compression |
| Refresh/banding | Strongest non-DC FFT component of row/column luminance means | 0.52 | Rolling shutter, LED light, blinds |
| Unusual sharpening | Mean absolute Laplacian relative to mean gradient | 0.58 | Camera sharpening, noise, facial hair |
| Resampling proxy | Maximum absolute autocorrelation at horizontal/vertical lags 2–6 | 0.52 | Natural repetition, denoising, interpolation |
| Planar glare/reflection | Fraction of near-white, low-chroma pixels | 0.55 | Oily skin, glasses, direct lights, clipping |
| Flat/axis edge behavior | Fraction of strong gradients close to image axes | 0.58 | Architecture, grid texture, orientation |

All scores are normalized engineering values in `[0,1]`, not probabilities. For
each family, 75% of the median outside-face score is subtracted from the facial
median. The maximum family score after taking the median over all outside regions
and frames must be ≤ 0.45. Camera/scene-wide banding and patterns therefore reduce
evidence in the controlled fixtures. This control cannot handle every spatially
local sensor or room artifact and can also suppress a real display artifact.

At least three of five facial regions must cross a family gate. At least two of
seven families must cross their gates in 75% of frames. The window requires six
samples spanning 450 ms. These requirements avoid treating any single artifact as
universal or sufficient. The seven measurements are correlated because they use
the same pixels and transforms; two families are not two independent security
factors.

Background flow must be available throughout the window. Face-center drift or
composed common scene motion over 3 px suppresses suspicious evidence because
motion blur and changing patch contents can alter spectra. Missing flow reduces
reliability. This is a conservative gate and can miss moving handheld attacks.

| Observation | Liveness confidence / reliability | Availability |
| --- | --- | --- |
| Short, unsupported or low-texture observation | 0.50 / 0 | False |
| Usable observation without persistent multifamily artifacts | 0.50 / 0.10 | True |
| Missing scene-flow context or excessive motion | 0.50 / 0.03 | True after minimum window |
| Persistent multifamily, multiregion candidate | 0.35 / 0.25 | True |

The candidate score is weak suspicion on a liveness-oriented scale, not a spoof
probability. It does not change the overall confidence from zero and cannot return
SPOOF. Absence of an artifact remains neutral at 0.50 and supplies no live credit.
No challenge or camera/display flicker test is issued.

Private history stores timestamp, 5 × 7 face scores, 4 × 7 background scores,
patch width/contrast, face center and scene-motion context for at most 64 samples
and two seconds. It stores no patch, frame, mask, FFT, autocorrelation, edge map or
raw landmark vector. Gaps over 350 ms reset it. Track changes, face loss, quality
failure, stale capture, discontinuity, exceptions, cancellation and terminal
failure clear it. Polling trims old samples and enforces freshness. A clean fresh
finish retains only historical scalar diagnostics, with availability false and
reliability zero; all private summaries are released. Cleanup failure clears even
that summary.

JSON exposes only counts and scalar family/background/persistence/motion diagnostics.
Strict worker IPC checks the known schema, finite ranges, score cap, STABLE quality
context, minimum patch/window support, multifamily/region/persistence conditions,
background threshold and motion gates. Unknown vectors, malformed booleans, excess
credit and LIVE/SPOOF are rejected.

## How to test on Fedora GNOME

From the project directory with the existing virtual environment and verified model:

```bash
.venv/bin/python tools/smoke_replay.py
.venv/bin/python tools/smoke_replay.py --json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Use a capture node reported by doctor and the ordinary Fedora desktop user. Add
`--preview` without `--json` only when Tk and a GNOME display are available. Try
real users, printed photos, phones/tablets/laptops at multiple resolutions and
distances, prerecorded video, moved media, glare, LED lighting, patterned clothing
and backgrounds. Record scalar JSON at attempt level if needed; raw frame retention
stays disabled. Every result remains non-success. Follow [evaluation.md](evaluation.md)
and keep held-out users, rooms, cameras and media separate from threshold work.

The offline tool opens no webcam and produces thirteen controlled cases:

| Generated case | Expected result |
| --- | --- |
| Natural random texture | Neutral 0.50 / 0.10 |
| Compound display luminance/chroma grid | Weak suspicion 0.35 / 0.25 |
| Achromatic grid standing in for halftone | Weak suspicion 0.35 / 0.25 |
| Camera-wide banding | Neutral because outside controls also respond |
| Background-only grid | Neutral because the face does not respond |
| One localized glare patch | Neutral; insufficient region/family support |
| Global sharpening | Neutral in the controlled image |
| Low texture | Unavailable 0.50 / 0 |
| Missing scene-motion context | Neutral 0.50 / 0.03 |
| Camera motion context | Neutral 0.50 / 0.03 |
| Sampling gap | Reset and unavailable |
| Two-frame transient display pattern | Neutral; insufficient persistence |
| Same display-like pattern labeled bona fide | Same weak suspicion, origin unresolved |

These are mathematical pixel fields, not rendered/photographed displays, prints,
skin or camera sensor simulations. The matching-bona-fide row deliberately shows
a false-positive/origin limit: identical pixels must produce identical diagnostics.
The fixture labels do not provide liveness ground truth. A complete generated
session runs the real tracking, motion and strict IPC path, then finishes UNCERTAIN
with confidence zero, historical replay scalars unavailable, and zero retained
tracking/private history. Tool exit 0 means engineering verification only.

## Verification performed

Environment: Fedora 44 Workstation, GNOME Wayland, x86_64, Python 3.14.7. Existing
MediaPipe, NumPy, OpenCV and the verified model were reused. No new dependency or
system package was added.

| Check | Result |
| --- | --- |
| Full automated suite | 184 tests passed in 89.107 s, including 20 new replay/ROI/session tests |
| Generated replay demo | Thirteen cases passed; complete session remained UNCERTAIN/confidence 0 |
| Generated replay JSON | All records finite scalar JSON; terminal result passed strict IPC validation |
| Real model initialization, blank inference and cleanup | Passed; zero faces, no webcam opened |
| Dependency consistency | `pip check`: no broken requirements |
| Doctor | Milestone 7, authentication disabled, no visible camera, Tk unavailable |
| Missing-camera capture | UNCERTAIN, confidence 0, zero retained history/replay evidence, exit 2 |
| Physical users, prints, displays, cameras and GNOME preview | Not run: no camera visible; Tk unavailable |

The 20 new tests cover equal-size inside/outside-face region support, background
separation, feature bounds, compound display/print patterns, natural texture,
global banding, background-only pattern, glare, sharpening, low texture, short and
transient observations, origin ambiguity, missing/moving scene context, every
track/time/reset boundary, bounded history/expiry, malformed frames/landmarks/time,
native update/poll/finish errors, cleanup failure, scalar JSON at 12/30 fps,
post-inference timeout, strict IPC and unsupported LIVE/SPOOF. All earlier capture,
tracking, motion, blink, gaze, illumination and watchdog tests pass.

## Known failures and future work

- High-quality displays and prints can contain none of the measured artifacts.
  Focus, distance, sensor resolution, compression and motion blur can erase them.
- Real skin, hair, glasses, clothing, walls, blinds, LED lighting and camera image
  processing can create the same periodic, edge, banding, sharpening or glare scores.
  Background controls and multifamily persistence do not eliminate false positives.
- Fixed face patches do not segment skin or detect a presentation-medium boundary.
  A large screen can cover background controls; a small/local camera artifact can
  occur only over the face. Static pose and landmark plausibility do not solve this.
- Fourier/autocorrelation features are scale and orientation dependent. Resizing to
  32 × 32 can create/erase periodicity; thresholds are uncalibrated for real cameras.
- The score families, landmarks, quality and background motion are dependent.
  Milestone 8 must model dependence and retain each individual score. It may not
  convert one replay candidate into SPOOF or one clean texture result into LIVE.
- No APCER, BPCER, false-positive, false-negative, demographic, camera-specific,
  CPU or latency result has been measured. Synthetic checks cannot supply them.

Milestone 8 remains planned. Physical data and dependence-aware fusion design must
precede any decision policy or authentication integration.

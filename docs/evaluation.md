# Separate manual capture and attack evaluation

Use the standalone harness as an ordinary Fedora GNOME or KDE Plasma desktop user. Camera frames
are not saved unless explicitly requested. Optional preview helps inspect mesh
placement but is not proof of liveness. Test with consent; keep any chosen debug
images local and delete them after review.

Today's procedure:

1. Run `--doctor`, then select an available capture node.
2. Run `.venv/bin/python -m facepass --device /dev/video0 --duration 5 --preview`
   (omit `--preview` when Tk is unavailable).
3. Check face count, landmark placement, quality reasons, tracking state,
   head-motion/geometry confidence and reliability, approximate pose, background
   availability, planar residual, blink phase/count/EAR/availability,
   gaze coordinates/coherence/agreement/head suppression, illumination ambient
   RGB balance/regional shading/spatial response/coherence/motion suppression,
   replay family scores/persistence/background control/motion suppression, temporal
   group support/overlap/agreement/conflicts, fusion coverage/penalty, observation
   status/limits/extensions and final
   non-success. Naturally blink when it happens; no deliberate action is required.
4. Repeat each scenario below. Record scalar JSON via `--json` if needed; parent
   output excludes the preview image and raw geometry.

| Scenario | Expected in milestone 9 / inspect | Future PAD evaluation |
| --- | --- | --- |
| Real user naturally looking at laptop | ACQUIRING → STABLE; inspect subtle pose span and weak motion scores after six frames/450 ms; never LIVE | Passive acceptance without deliberate action |
| Real user unusually still | Tracking can stabilize; motion scores stay neutral, never automatic SPOOF | Acceptance from adequate diverse evidence, otherwise UNCERTAIN |
| Real user with glasses | Inspect eye/iris contours, blink/gaze availability and pose quality; unreliable geometry clears eye evidence, but glasses/occlusion are not directly classified | Reduced eye evidence reliability without arbitrary rejection |
| Low light | Underexposure/noise/landmark failure may reduce quality | Usually insufficient quality, not arbitrary SPOOF |
| Bright backlight | Inspect face exposure, not background mean | Insufficient quality when face is unusable |
| Side lighting | Inspect landmark stability and uneven shading | Avoid penalizing plausible asymmetric lighting |
| Printed photograph | Inspect periodic/halftone, resampling, sharpening and edge diagnostics; no visible artifact stays neutral | Detection and false-negative rate by printer/paper/distance |
| Phone displaying photo | Inspect periodic/chroma-grid/banding/glare diagnostics; weak suspicion requires persistent multifamily/region support | Detection by display technology, resolution, brightness and distance |
| Phone displaying prerecorded video | Internal face movement, blinks and gaze can gain weak scores; still never LIVE | Replay false acceptance |
| Laptop/tablet replaying video | Test several screen sizes/resolutions | Replay false acceptance |
| Spoof medium moved/tilted | Inspect held-out planar error and geometry noise; a fitted plane should not gain positive scores, but real-model artifacts can confound it | Planar motion/perspective confounders |
| Multiple faces | Terminal UNCERTAIN, multiple_faces, empty history, if both detected | Ambiguity refusal, including small second faces |
| Face enters/leaves frame | Clear history on loss, new track number and ACQUIRING on return | Reset all temporal evidence; reject face switching |
| Camera unavailable mid-session | UNCERTAIN or watchdog failure; camera released | Fail-closed termination |
| Malformed/failed frames | Automated tests: terminal non-success | Fail-closed termination |
| Brightness-adjusted replay | Still never LIVE | Lighting/display score robustness |
| Replay in similar lighting | Still never LIVE | Coherent replay false acceptance |
| Different face replaces subject while visible | Strong shape/appearance/position change may terminate UNCERTAIN | Similar-face substitution needs separate identity checks |
| Fast camera/head movement | May conservatively terminate UNCERTAIN on discontinuity | Measure usability costs separately from attack rejection |
| Small camera/whole-scene movement | With usable background texture, inspect common-motion compensation and neutral scores; no exact pixel compensation is promised | Parallax, sensor motion and interpolation confounders |
| Textureless wall or blocked background | Background motion becomes unavailable; usable pose alone does not gain a positive score | Reduced motion evidence without arbitrary SPOOF |
| Frame gap without complete failure | Over 220 ms clears blink/gaze; over 350 ms clears illumination/replay/tracking history; fresh observation required | Never join evidence across missing observations |
| Natural bilateral blink | Inspect open → closing → closed → opening → open and one count only after confirmed reopening; weak score at most 0.65/0.35 | Measure physical sensitivity across people, anatomy and cadence |
| No blink during observation | Usable open eyes stay neutral, never automatic SPOOF | Passive acceptance using other adequate evidence |
| Eyes initially or continuously closed | No blink credit; sustained closure distinguished from an established cycle | Visibility/usability effects without arbitrary spoof rejection |
| Wink, squint or noisy single closure | No credit for unconfirmed/asymmetric sequence; blink may become unavailable | Separate genuine asymmetry from landmark noise |
| Fast blink / low processing cadence | May miss closure confirmation; do not bridge gaps or lower thresholds | Measure missed sequences and CPU/cadence tradeoffs |
| Small, occluded or reflected eyes | Inspect EAR plausibility and availability; no calibrated visibility guarantee | Occlusion-related false sequences and nonresponse |
| Natural eye movement with a nearly still head | Inspect relative iris displacement, bilateral agreement and coherence; six samples/450 ms minimum, weak score ≤ 0.62/0.30 | Physical sensitivity and missed nonlinear movements |
| Fixed gaze / no eye movement | Usable eyes stay neutral, never automatic SPOOF; no direction requested | Acceptance through other adequate evidence |
| Eye movement while turning the head | Head span over 1.5° suppresses gaze positives; inspect conservative neutral reliability | Genuine vestibulo-ocular/head coupling and perspective confounders |
| One eye visible / iris too small | Gaze unavailable with zero reliability; usable face track can remain STABLE | Partial visibility, anatomy and glasses-related nonresponse |
| Static side light / unequal face and background colors | Regional shading and RGB balance reported; no illumination positive from static asymmetry or brightness difference | Skin/room reflectance diversity and mixed lighting |
| Uniform exposure or common white-balance changes | Common log-RGB gain removed; remaining illumination evidence neutral | Camera pipelines, automatic controls and local tone mapping |
| Naturally changing spatial room light | Inspect seven face regions/four ambient tiles, at least eight samples/600 ms, direction/coherence/color error and weak score ≤ 0.58/0.20 | Physical response sensitivity; no lighting challenge required |
| Ambient change without facial response or with opposed RGB response | Mismatch diagnostic, neutral confidence 0.50 with reliability 0.05 when context is usable; never automatic SPOOF | Genuine geometry/reflectance exceptions and adaptive replay |
| Lighting change during head or camera movement | Head span > 1.5° or face/background movement > 2 px suppresses illumination positives | Parallax, changing ROI contents and usability costs |
| Missing background flow or insufficient ambient tiles | Missing flow reduces neutral reliability; missing tiles makes illumination unavailable | Scene texture, close framing and moving objects |
| Face patches obscured by glasses, beard or reflection | Inspect region support/clipping; geometry can remain plausible despite occlusion | Patch/anatomy/reflectance confounders and false positive response |
| Room-lit print or replay matching lighting response | Can earn weak illumination score, still never LIVE | Physical false acceptance; regional residual is not depth proof |
| Real skin, facial hair, patterned clothing or glasses | Inspect texture scores and region support; natural repeating texture can produce weak suspicion | Bona-fide false positives across appearance groups |
| Blinds, LED lighting or camera rolling shutter | Background control should keep camera/scene-wide banding neutral when sampled; still never automatic SPOOF | Camera-wide vs face-local artifact discrimination |
| High-resolution/retina display or quality print | Missing artifacts stay neutral, never positive liveness evidence | Physical replay false negatives |
| Local glare/reflection | One affected patch/family remains neutral in controlled tests | Skin/glasses/display glare false positives and missed combined attacks |
| Moving replay or handheld camera | Face/common motion over 3 px suppresses replay suspicion | Motion blur, changing patch content and usability costs |
| Textureless or too-small face patches | Replay producer unavailable with zero reliability | Distance/resolution nonresponse rate |
| Aligned motion and illumination | At least eight stable samples can show two-group overlap and weak temporal support; global confidence remains zero | Cross-signal calibration and held-out bona-fide/attack rates |
| Only landmark or only illumination support | Temporal summary remains neutral with a missing group | Measure usable nonresponse without rewarding correlated evidence |
| Replay suspicion overlaps positive groups | Temporal conflict is explicit; no SPOOF candidate and no positive temporal signal | Evaluate adaptive/coherent replay false acceptance |
| Sampling gap or stale capture | Temporal window resets and cannot join evidence across the gap | Recovery/usability and reset behavior |
| Stable two-group support after the minimum window | May finish early with `UNCERTAIN` confidence 0; inspect readiness and disabled challenge state | Early-completion usability and held-out attack/nonresponse rates |
| Partial conflict-free evidence at base limit | At most one bounded extension, then hard timeout | Extension benefit versus user burden and attack exposure |
| Evidence conflict at base limit | No extension; hard timeout remains fail-closed | Conflict frequency and false timeout rate |

Remove the camera only when safe for the hardware, or close a removable webcam's
connection during observation. A built-in camera can instead be tested using the
simulated camera-failure test and a deliberately unavailable capture node. Do not
modify system authentication or device permissions to simulate failure.

Run `.venv/bin/python tools/smoke_motion.py --json` first for generated examples of
these scalar fields. The generated 3D rotation uses the same idealized template
and intrinsics as the pose estimator. It verifies small-motion math, not real face
accuracy. Motion scores use a two-second baseline; the six-frame/450-ms minimum
does not authorize early authentication. At a clean finish, only historical scalar
motion/blink/gaze/illumination/replay/temporal diagnostics remain historical with live
availability false and reliability zero. Every
loss/quality/stale/error/cancel path must clear those diagnostics and caches.

Run `.venv/bin/python tools/smoke_blink.py --json` for eight generated blink/noise/gap
cases. These change idealized eye landmarks over static generated pixels; they do
not simulate physical eyelid video. The identical replay pattern deliberately gains
the same weak score as the generated bilateral blink. Automated sequence tests
cannot measure attack false acceptance or natural blink sensitivity. Check real
video/media separately; keep raw capture disabled unless explicitly needed.

Run `.venv/bin/python tools/smoke_gaze.py --json` for thirteen generated iris-motion,
noise, quality and head/scene confounder cases. These modify landmarks over static
pixels, not physical eye video. An identical replay pattern earns the same weak
score. Real-user eye movement, glasses, occlusion, head coupling and diverse replay
media require separate testing. Do not infer physical gaze or attack accuracy from
these fixtures, or lower gates merely to obtain a positive signal.

Run `.venv/bin/python tools/smoke_illumination.py --json` for fifteen controlled
pixel-gain scenarios. These test static shading, common exposure/WB removal,
spatial/color response, planar response, tiny changes, missing scene context,
head/camera movement, a gap and an identical replay. The replay earns the same weak
score as the spatial-response fixture. The patches and gains are authored to
exercise the measurement logic; they are not physical skin, 3D lighting or display
simulations. Standalone cases supply known synthetic motion context, while the
complete session runs the actual background-flow analyzer on generated pixels.
Evaluate real rooms/cameras/users/media separately without retaining frames by
default. Do not require deliberate light changes to obtain a usable capture.

Run `.venv/bin/python tools/smoke_replay.py --json` for thirteen controlled texture
cases. Generated display-grid and halftone fields receive weak suspicion; natural
texture, global banding, background pattern, local glare, global sharpening,
transient artifacts, missing context and gaps are neutral/unavailable as applicable.
The matching-bona-fide-pattern case receives the same suspicious score as the
display fixture, intentionally showing the false-positive/origin limit. These are
pixel patterns rather than photographed phones, screens or prints. Do not infer
APCER/BPCER or tune thresholds from them; collect physical attempt-level data.

Run `.venv/bin/python tools/smoke_temporal.py --json` for generated landmark-only,
photometric-only, aligned-two-group, replay-conflict, missing-context and sampling-gap
cases. The smoke output is a scalar dependence audit; it is not a calibrated fusion
model or an attack benchmark.

Run `.venv/bin/python tools/smoke_observation.py --json` for collecting,
early-completion, one-extension, hard-timeout and conflict-timeout policy cases.
These scenarios exercise duration control only; they do not measure physical user
completion or attack performance.

## Evaluation design and attempt recorder

The implemented [evaluation tooling](evaluation-tooling.md) records scalar labelled
attempts and reports these denominators by source/policy/configuration/split/scenario.
Run `python -m facepass.evaluation capture --help` and
`python -m facepass.evaluation report --help` in the installed environment.

Record attempts as units, including failures and timeouts. Label ground truth as
`bona_fide` or `attack`, and record a non-identifying scenario code, quality, signal
availability, final decision, elapsed time and model/config version. Vary users,
skin tones, glasses, facial hair, cameras, rooms, lighting and print/display media.
Reserve users, rooms and attack media for held-out evaluation; do not tune only to
one person's room or one display. Report per-condition counts and uncertainty, not
just an aggregate success percentage.

- False acceptance: attack attempts returning LIVE / all attack attempts. Also
  report results restricted to usable captures so quality failures cannot hide a
  weak PAD detector.
- False rejection: bona fide attempts explicitly returning SPOOF / all bona fide
  attempts. Separately report bona fide UNCERTAIN and INSUFFICIENT_QUALITY rates,
  plus total non-acceptance (every decision other than LIVE).
- If reporting PAD APCER/BPCER, define the treatment of non-responses and quality
  failures explicitly; do not label every capture failure a classifier rejection.
- Track latency distribution and cancellation delay separately from decisions.

The current disabled decision policy supplies no meaningful PAD false-acceptance/false-rejection estimate:
success is disabled by construction. The automated tests and blank-frame smoke
check are engineering verification, not evidence of attack resistance. Hardware
scenarios remain untested in this environment because no camera is visible. Head
motion and geometry share the same model and measurements; score agreement must
not be counted as independent evidence of attack resistance. Illumination also
reuses landmarks, head pose and background tracking; agreement with their scores
must not be treated as independent validation.
Replay features share pixels, regions and background motion, and their seven family
scores are correlated rather than seven independent security votes.

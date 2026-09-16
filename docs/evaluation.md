# Separate manual capture and attack evaluation

Use the standalone harness as an ordinary Fedora GNOME desktop user. Camera frames
are not saved unless explicitly requested. Optional preview helps inspect mesh
placement but is not proof of liveness. Test with consent; keep any chosen debug
images local and delete them after review.

Today's procedure:

1. Run `--doctor`, then select an available capture node.
2. Run `.venv/bin/python -m facepass --device /dev/video0 --duration 5 --preview`
   (omit `--preview` when Tk is unavailable).
3. Check face count, landmark placement, quality reasons, tracking state,
   head-motion/geometry confidence and reliability, approximate pose, background
   availability, planar residual and final non-success.
4. Repeat each scenario below. Record scalar JSON via `--json` if needed; parent
   output excludes the preview image and raw geometry.

| Scenario | Expected in milestone 3 / inspect | Future PAD evaluation |
| --- | --- | --- |
| Real user naturally looking at laptop | ACQUIRING → STABLE; inspect subtle pose span and weak motion scores after six frames/450 ms; never LIVE | Passive acceptance without deliberate action |
| Real user unusually still | Tracking can stabilize; motion scores stay neutral, never automatic SPOOF | Acceptance from adequate diverse evidence, otherwise UNCERTAIN |
| Real user with glasses | Inspect eye tracking, pose reliability and quality; gaze/blink still unavailable | Reduced gaze reliability without arbitrary rejection |
| Low light | Underexposure/noise/landmark failure may reduce quality | Usually insufficient quality, not arbitrary SPOOF |
| Bright backlight | Inspect face exposure, not background mean | Insufficient quality when face is unusable |
| Side lighting | Inspect landmark stability and uneven shading | Avoid penalizing plausible asymmetric lighting |
| Printed photograph | A mesh and acceptable quality are possible; missing motion stays neutral | Static print rejection or uncertainty |
| Phone displaying photo | A mesh and acceptable quality are possible; display detector is unavailable | Display/planar evidence |
| Phone displaying prerecorded video | Internal face movement can gain weak scores; still never LIVE | Replay false acceptance |
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
| Frame gap without complete failure | Over 350 ms clears history; fresh acquisition required | Never join evidence across missing observations |

Remove the camera only when safe for the hardware, or close a removable webcam's
connection during observation. A built-in camera can instead be tested using the
simulated camera-failure test and a deliberately unavailable capture node. Do not
modify system authentication or device permissions to simulate failure.

Run `.venv/bin/python tools/smoke_motion.py --json` first for generated examples of
these scalar fields. The generated 3D rotation uses the same idealized template
and intrinsics as the pose estimator. It verifies small-motion math, not real face
accuracy. Motion scores use a two-second baseline; the six-frame/450-ms minimum
does not authorize early authentication. At a clean finish, only historical scalar
motion diagnostics remain, with availability false and reliability zero. Every
loss/quality/stale/error/cancel path must clear those diagnostics and caches.

## Evaluation design for later milestones

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

Milestones 1–3 supply no meaningful PAD false-acceptance/false-rejection estimate:
success is disabled by construction. The automated tests and blank-frame smoke
check are engineering verification, not evidence of attack resistance. Hardware
scenarios remain untested in this environment because no camera is visible. Head
motion and geometry share the same model and measurements; score agreement must
not be counted as independent evidence of attack resistance.

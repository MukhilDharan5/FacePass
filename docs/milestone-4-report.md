# Milestone 4 — optional bilateral blink sequences

Version 0.4.0 adds a conservative landmark-based blink sequence detector to the
standalone Fedora GNOME harness. It estimates eyelid closure/reopening over time;
it cannot establish whether the movement belongs to a live person or a replay.
Blinking remains optional. Overall confidence stays zero, LIVE/SPOOF fusion stays
disabled and every camera run exits 2. No authentication configuration was changed.
Development stops here before milestone 5 (gaze movement).

## Implementation

`blink.py` provides a separate `BlinkAnalyzer`, fed the current immutable sample
from an uninterrupted, quality-accepted STABLE track. Six MediaPipe landmarks per
eye yield pixel-space eye aspect ratio (EAR), preserving image aspect ratio and
tolerating rigid roll/uniform scaling. The detector keeps only scalar phase state,
timers/counters and two previous eye widths; it retains no new camera frames or raw
eye trajectories. Existing tracker history remains bounded at 64 samples/two seconds.

```text
ACQUIRING_OPEN → OPEN → CLOSING → CLOSED → OPENING → OPEN
                                  └─ prolonged closure → SUSTAINED_CLOSED
```

An open baseline must be confirmed before a cycle begins. Both eyes must supply
at least two consecutive closed samples spanning 60 ms, then two consecutive open
samples spanning 60 ms. Reopening confirmation must arrive within 650 ms of the
first below-open sample. Intermediate closing/opening motion may fall between
camera samples, but a single closed frame is never enough. Initially closed eyes,
held closure, separated noise spikes and an interrupted reopening do not count.

Initial thresholds are EAR ≥ 0.23 for open and ≤ 0.15 for closed, with hysteresis
between them. Eye width must be at least 18 px. Out-of-frame, crossed/displaced,
implausible, asymmetric or suddenly resized eye contours clear blink state and
make its evidence unavailable. These checks do not measure calibrated visibility
or detect every occlusion. Detailed geometry gates are in [architecture.md](architecture.md).

| Observation | Confidence | Reliability | Availability |
| --- | --- | --- | --- |
| Short/unreliable eye observation | 0.50 | 0 | False |
| Usable eyes, no completed fresh blink | 0.50 | 0.15 | True after three samples/150 ms |
| Completed bilateral sequence, reopened and fresh | 0.65 | 0.35 | True |

Scores are uncalibrated engineering values. More blinks never increase their caps.
Positive credit lasts at most two seconds while the phase remains OPEN. Any gap
over 220 ms clears the blink state and credit, even before the track's 350-ms
expiry. `current_result()` enforces cadence and credit expiry without a new frame.
No sample interpolation or threshold lowering compensates for missing observations.

Session integration resets blink state on track change, face loss, poor quality,
staleness, discontinuity, multiple faces, native/detector exceptions, failure and
cancellation. A clean fresh finish may retain historical scalar diagnostics, with
availability false/reliability zero, after releasing private state. Finishing during
closure does not complete a blink. Cleanup failure clears even historical values.

Terminal output, preview and JSON expose blink confidence/reliability/availability,
phase, current eye measurements, counts and timing. Strict worker validation rejects
unknown/vector fields, NaN, invalid types, inconsistent evidence context, excessive
scores and positive evidence without fresh confirmed reopening. The existing worker
supervision and disabled-success behavior remain in force. Gaze, illumination,
replay artifacts and temporal consistency are still unimplemented.

## Assumptions and known failures

- The local OS, camera stream and model are trusted. Stable tracking is heuristic
  continuity, not identity recognition or a guarantee against a face substitution.
- EAR thresholds, eye-size/bilateral gates and timing limits are initial, fixed
  engineering choices. They are not personalized or calibrated across users.
- At the default 12 processed fps, fast natural blinks may lack two sampled closed
  frames. Even high cadence cannot credit a closure shorter than the sampled
  confirmation span. Missed/no blink stays neutral, never automatic SPOOF.
- Glasses, reflections, eye anatomy, partial occlusion, perspective and landmark
  noise can cause missed or false sequences. Plausible model output is not proof
  that the eye was physically visible. A conservative asymmetry gate can discard
  genuine uneven closure; sustained closure may reflect discomfort or fatigue.
- A prerecorded natural blink can earn exactly the same weak score. Coherent
  generated animation or model hallucination on flat media may also imitate it.
  Blink and motion share the landmark model and are dependent evidence; future
  fusion must not count them as independent security factors.
- Synthetic landmark fixtures verify geometry and sequencing over static generated
  pixels. They do not test real eyelid image inference, physical replay resistance,
  PAD false acceptance/rejection, CPU usage or authentication latency.

The EAR measurement follows [Soukupová and Čech's paper](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-CVWW-2016.pdf).
This implementation uses its own state machine; it does not implement the paper's
trained classifier or inherit its evaluation results. Eye contour indices were
checked against [MediaPipe's contour definitions](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py).

## Test on Fedora GNOME

From the project directory, after the existing environment/model setup:

```bash
.venv/bin/python tools/smoke_blink.py
.venv/bin/python tools/smoke_blink.py --json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m facepass --doctor
.venv/bin/python -m facepass --device /dev/video0 --duration 5 --json
```

Select a capture node reported by doctor. To inspect mesh placement, omit `--json`
and add `--preview` when Fedora's `python3-tkinter` and a GNOME display are available.
Look naturally at the laptop; no deliberate blink or head turn is required. Check
the individual signal output and final non-success. Raw image retention stays off
unless the existing explicit diagnostic option is supplied.

The offline demo uses no webcam and prints these expected outcomes:

| Generated scenario | Blink count | Confidence | Result |
| --- | --- | --- | --- |
| Bilateral closure/reopening | 1 | 0.65 | Weak sequence evidence |
| Open stillness | 0 | 0.50 | Neutral |
| Sustained closure | 0 | 0.50 | SUSTAINED_CLOSED, neutral |
| Single-frame collapse | 0 | 0.50 | Unconfirmed sequence rejected |
| Asymmetric eyes | 0 | 0.50 | Sequence reset; brief final observation unavailable |
| Low eye resolution | 0 | 0.50 | Unavailable |
| Sequence crossing a 300-ms gap | 0 | 0.50 | Reset; no joined blink |
| Identical prerecorded-pattern input | 1 | 0.65 | Same score, demonstrating the origin limitation |

The replay row repeats the same generated geometry; it is not a physical screen
test. The complete generated session finishes UNCERTAIN, confidence 0, blink values
marked historical, signals unavailable and retained history zero. The demo exits 0
for engineering verification, not authentication.

## Verification performed

Environment: Fedora 44 Workstation, GNOME Wayland, x86_64, Python 3.14.7. Editable
package metadata was updated to 0.4.0. The existing MediaPipe/NumPy/OpenCV dependencies
and verified model were reused; no new runtime dependency or system package was needed.

| Check | Result |
| --- | --- |
| Full automated suite | 103 tests passed, including 24 new blink/session tests |
| Generated blink demo and scalar JSON | Eight scenarios passed; final result passed strict worker validation |
| Existing motion demo | Passed; previous weak/neutral outcomes and final non-success preserved |
| Real model initialization, blank inference and cleanup | Passed; zero detected faces, no webcam opened |
| Dependency consistency | `pip check`: no broken requirements |
| Doctor | Milestone 4, authentication disabled, no visible camera, Tk unavailable |
| Missing-camera JSON capture | UNCERTAIN, confidence 0, no retained blink/motion/history, exit 2 |
| Physical blinks, prints, displays, replays and GNOME preview | Not run: no camera visible; Tk unavailable |

New tests cover pixel-space EAR and roll/scale invariance, full/initially closed/
held/incomplete sequences, single-frame and separated spikes, hysteresis, asymmetry,
small/clipped/crossed/displaced contours, geometry jumps, minimum closure duration,
cadence boundaries, track changes, positive-credit expiry during updates and polls,
repeated-blink caps, malformed timestamps/NaN/native exceptions, historical finish,
all session reset/cleanup paths and invalid worker evidence. Earlier capture,
tracking, motion and watchdog tests continue passing.

Physical evaluation should follow [evaluation.md](evaluation.md), including real
users, stillness, glasses, lighting, timing/cadence and replay media. Hardware
evidence must guide later calibration; passing these tests does not establish
attack resistance. Milestone 5 remains planned and has not started.

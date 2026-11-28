# Incremental implementation plan

The request to complete the project in one pass supersedes the earlier per-milestone
pause. Software for the roadmap is implemented; physical evaluation remains open.
PAM, GDM/SDDM, GNOME/KDE authentication and `/etc/pam.d` stay outside scope.

| Milestone | Scope and acceptance criteria | Status |
| --- | --- | --- |
| 1 | Local camera/landmarks, bounded diagnostic harness, quality and non-success failures | Implemented; hardware/preview testing outstanding |
| 2 | Stable one-face tracking; detect discontinuities and clear history/evidence; no identity/liveness claim | Implemented; hardware face-switch testing outstanding |
| 3 | Subtle head/landmark motion and planar/background motion comparison; still people need no challenge | Experimental implementation complete; physical motion/attack evaluation outstanding |
| 4 | Blink sequence state machine, noise/failure discrimination, no mandatory blink | Experimental implementation complete; physical blink/occlusion/replay evaluation outstanding |
| 5 | Iris/gaze changes relative to head with explicit resolution/visibility reliability | Experimental implementation complete; physical eye/head/glasses/replay evaluation outstanding |
| 6 | Multi-region face shading/color vs ambient scene and temporal lighting response | Experimental implementation complete; physical lighting/camera/reflectance/attack evaluation outstanding |
| 7 | Replay/print texture and frequency evidence with confounder controls | Experimental implementation complete; physical camera/media false-positive and detection evaluation outstanding |
| 8 | Temporal consistency and fusion retaining individual scores and dependence groups | Experimental implementation complete; physical cross-signal calibration outstanding |
| 9 | Adaptive observation with minimum window, conflict-aware early completion, one bounded extension and hard timeout | Experimental implementation complete; physical usability and policy calibration outstanding |
| 10 | Attack evaluation across the full scenario matrix and held-out conditions | Recorder/report tooling complete; physical study and calibration outstanding |
| 11 | Profile/optimize CPU, detector cadence and bounded cancellation | Benchmarks and capture profiles complete; extension deadlines fixed; physical workload measurements outstanding |
| 12 | Versioned local daemon interface; identity remains separate | Research-only Unix-socket observe/status/cancel service complete; authentication remains disabled pending evaluation |

Milestone 1 assumptions: ordinary V4L2 RGB webcam, trusted local OS/device/model,
one nearby face, CPU inference, 640 × 480 target, experimental quality thresholds.
No reliable depth, occlusion or liveness is established yet. Milestone 2
establishes heuristic observation continuity and bounded history, not identity.
Milestone 3 adds approximate pose and two weak, correlated motion evidence producers;
it does not enable fusion, SPOOF or LIVE. Milestone 4 adds conservative bilateral
EAR sequences and constant-memory phase state; no blink stays neutral. Its
thresholds are uncalibrated, fast blinks may be missed at 12 fps and prerecorded
blinks can earn the same weak score. Milestone 5 adds relative iris movement with
bilateral/window/pose gates and a weak score capped at 0.62/0.30. It does not
estimate calibrated gaze or measured eye visibility; missing gaze stays neutral.
Milestone 6 adds seven facial and four outside-face RGB region summaries, temporal
spatial response, common exposure/WB gain removal and motion-confounder gates.
Its weak score is capped at 0.58/0.20. Ambient RGB balance and response direction
are uncalibrated image measurements, not Kelvin temperature, actual emitter
direction or measured depth. Uniform changes, mismatches and missing evidence
stay neutral/unavailable; a matching replay can earn the same score.
Milestone 7 adds five face/four background texture patches, seven artifact-family
diagnostics and temporal/background/motion gates. Persistent evidence from at least
two families across three regions produces only weak suspicion at liveness confidence
0.35/reliability 0.25. Absence of an artifact stays neutral rather than supplying live credit;
camera-wide patterns are controlled using background patches. These uncalibrated
metrics do not identify a physical print/display and cannot directly return SPOOF.
Milestone 8 adds a bounded 64-sample/two-second scalar window,
landmark/photometric/replay dependence groups, cross-group overlap and agreement,
explicit replay conflicts, and a transparent dependence penalty. The temporal signal
is capped at 0.56/0.18 and remains diagnostic; global confidence stays zero and both
fusion candidate flags are disabled.
The model
does not expose calibrated per-mesh confidence, and no camera or Tk preview could
be tested here. Milestone 9 adds a 1.5-second minimum observation window, conflict-aware
early completion only after stable two-group support, at most one two-second extension
for partial evidence, a hard timeout, and an explicitly disabled challenge state.
The policy controls capture duration only; it cannot produce LIVE/SPOOF. See README
for commands and evaluation.md for manual cases.

See the [completion report](completion-report.md), [evaluation tooling](evaluation-tooling.md)
and [daemon protocol](daemon-protocol.md) for milestones 10–12, commands and verification.
No physical PAD accuracy or production authentication readiness has been established.

Future milestones must continue returning non-success until the quality, tracking
and fusion policy have enough validated evidence. Implementing a blink or a motion
detector alone is never permission to enable LIVE. A challenge cannot replace
identity recognition or bypass fusion/quality policy.

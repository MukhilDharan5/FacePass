# Software completion report — 11 October 2026

This report records the original research delivery. Version 0.14.0 subsequently
adds the separate experimental authenticator described in [face-unlock.md](face-unlock.md).
Its current regression results are in [auth-verification-reference.jsonl](auth-verification-reference.jsonl).

Version 0.12.0 implements the remaining software roadmap on top of milestones 1–9.
The earlier per-milestone pause was superseded by the request to complete the
project in one pass. The project remains a local RGB research engine. Real-world
presentation-attack resistance, calibrated fusion, physical user testing and
production authentication have not been established. No `/dev/video*` device is
available in this execution environment. Tk is installed; desktop preview with a
physical camera remains untested.

## Delivered changes

Milestone 10 supplies a scalar attempt recorder and strict JSONL/reporting workflow.
Reports show explicit all-attempt and usable-capture denominators, bona-fide
rejections versus nonresponses, availability, scenario counts, latency percentiles
and Wilson intervals. Physical/synthetic sources, policy/model/configuration
versions and splits remain separate. Duplicate records, unknown fields and
development/held-out participant, room and media leakage are refused. The collector
requires private file permissions, locks each append and never saves frames or
landmarks. See [evaluation-tooling.md](evaluation-tooling.md).

Milestone 11 adds bounded end-to-end capture timing/CPU summaries, synthetic full
detector benchmarks, optional real-model blank inference and optional cProfile
output. Profiling identified avoidable recursive copying during scalar diagnostic
validation; validation now reads dataclass fields directly, while public
serialization still makes owned dictionaries. Thread limits and full-frame face
detection cadence are preserved. Detector thresholds and evidence caps are unchanged.

The observation integration now honors an approved extension throughout both
frame processing and supervisor deadlines. Previously, the model-return check
still used the base deadline, and the supervisor allowed only the base plus 500 ms.
Both could cut off an extension. Fixed windows and one configurable 0–5 second
extension now use the same validated configuration everywhere. Counter bounds cover
35 seconds at 30 fps, and IPC refuses inconsistent duration limits or a second
extension. Startup, frame stalls, crashes and cancellation remain independently
bounded. The timing tools distinguish startup, active observation and synthetic
throughput; blank inference is not representative of a detected real face.

The [reference benchmark](benchmark-reference.json) records one 60-frame run on
Python 3.14.8/x86_64 with one OpenCV thread. The real-model blank-input p50/p95 were
1.695/2.164 ms; the synthetic complete-detector p50/p95 were 21.076/22.860 ms.
Real-model initialization took 897.432 ms. These workloads were unpaced, and the
synthetic pipeline used generated landmarks. They cannot predict physical
face-inference latency or attack performance. No speedup percentage is claimed
from one run or comparisons across profiling/instrumentation conditions.

Milestone 12 supplies a foreground, same-user Unix-socket diagnostic service and
client. Versioned observe/status/cancel requests wrap the existing supervised finite
worker. One capture at a time, bounded connections, strict options, peer UID checks,
private socket permissions, disconnect cancellation, bounded writes and graceful
shutdown preserve the engine's boundaries. Idle service operation opens no model
or camera. The service exposes no identity, enrollment, preview or image retention.
See [daemon-protocol.md](daemon-protocol.md).

Packaging includes the `facepass`, `facepass-evaluate`, `facepass-daemon` and
`facepass-client` entry points. A single verification runner and GitHub Actions
workflow exercise unit tests, local-socket integration, dependency consistency,
verified model initialization/blank inference and every generated scenario helper.
The workflow is provided for future repository runs; local verification is the
evidence collected in this report.

## Verification and remaining work

All 12 checks in the complete offline verification runner passed, including its
236-test unit/integration run. After the final partial-request deadline hardening,
all 12 daemon tests passed, including the additional slow-fragment regression.
The test collection at that point contained 237 tests. The complete verification
summary is saved in [verification-reference.jsonl](verification-reference.jsonl).
The subsequent KDE compatibility update adds 11 desktop tests; all 248 tests and
all 12 complete verification checks now pass with KDE session settings. The
latest run is saved in [kde-verification-reference.jsonl](kde-verification-reference.jsonl).
Read-only desktop diagnostics and preview failure guidance support Konsole and
Plasma Wayland while preserving GNOME/headless capture. A real Tk display
connection, hidden generated-PNG preview and cleanup also passed in an environment
reporting KDE Wayland. See [Fedora KDE setup and verification](fedora-kde.md).
The pinned model and real supervised daemon missing-camera path both passed.
Editable installation and wheel construction succeeded, and all four installed
command entry points were checked. `git diff --check` passed.

```bash
.venv/bin/python tools/verify_project.py
.venv/bin/python tools/profile_pipeline.py --frames 60 --model models/face_landmarker.task
.venv/bin/python -m facepass --device /dev/video999 --duration 0.5 --json --profile
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/facepass-wheel
```

The suite covers malformed frames/landmarks/JSON, quality loss and tracking resets,
detector confounders, temporal dependence, all terminal cleanup paths, unexpected
success, extension timing, record privacy and held-out leakage, daemon concurrency,
peer UID rejection, client disconnects and cancellation of a hung real subprocess.
Partial requests use an absolute deadline that does not reset on each new fragment.
The verified pinned model processes blank synthetic pixels offline. Missing-camera
capture terminates with UNCERTAIN, confidence 0 and capture status 2. The package
build and installed entry points are checked separately from detector correctness.

Physical webcam capture and webcam preview on GNOME/KDE, real-face inference latency, laptop thermal
load, held-out subjects/rooms/media, usable completion rates, clustered uncertainty
analysis and calibrated decision policy remain outstanding. The available tools
make that work recordable; they cannot perform it without hardware and a physical
study. No threshold tuning or authentication enablement is justified by these
synthetic tests. PAM, GDM/SDDM and desktop login/lock-screen integration remain
outside the original scope.

Every runtime observation still returns non-success. A zero attack-acceptance count
from a disabled policy is expected and is not evidence of PAD accuracy. This is
software completion of the research tooling, with the empirical validation gap
explicitly retained.

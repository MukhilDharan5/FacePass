# Milestone 8 — bounded temporal consistency and dependence-aware fusion

Milestone 8 is implemented for the Fedora GNOME standalone harness. It adds a
bounded scalar temporal layer over the milestone 3–7 evidence producers and makes
the dependence assumptions visible in the JSON result. It does not add PAM, GDM,
GNOME, identity recognition, enrollment, a service, or an authentication decision.

## Delivered behavior

`TemporalConsistencyAnalyzer` retains no frames, landmarks, embeddings, spectra or
patches. It keeps at most 64 timestamped scalar snapshots over two seconds and
clears them on a non-stable track, a sampling gap over 350 ms, tracking expiry,
quality loss, cancellation, failure and clean release. A duplicate or non-finite
timestamp raises a bounded evidence error and the session fails closed.

The analyzer uses three explicit dependence groups:

1. `landmark_dynamics`: head motion, facial geometry, blink and gaze. Their
   contribution is the strongest one normalized support, so correlated landmark
   measurements cannot be counted as four independent votes.
2. `photometric_scene`: multi-region illumination response, kept separate from
   landmark dynamics but still gated by scene/motion context.
3. `replay_texture`: replay-artifact suspicion, retained as a negative/conflict
   group and never treated as positive live support.

The result exposes each group’s availability and support, the number of positive
groups and supporting individual signals, temporal duration, cross-group overlap,
agreement, stability, missing groups, scene-motion suppression and replay/mismatch
conflicts. `FusionDiagnostics` reports group scores, coverage, the transparent
dependence penalty, raw and adjusted diagnostic support, and explicit
`liveness_candidate`/`spoof_candidate` flags. Both candidate flags are always false.

At least eight stable samples spanning 600 ms are required before the temporal
signal is available. One group or missing context remains neutral. Aligned support
from landmark dynamics and photometric scene evidence can produce only
`0.56 confidence / 0.18 reliability` with diagnostic text
`weak_dependence_aware_temporal_consistency`; replay overlap or response mismatch
forces a neutral conflict result. Overall `LivenessResult.confidence` remains zero
and its decision remains `UNCERTAIN` or `INSUFFICIENT_QUALITY`.

The strict worker validator checks the new schemas, scalar bounds, terminal
historical state, group counts, positive-evidence context and disabled candidate
flags. Individual signal slots remain in every result so a future calibration can
inspect which evidence was actually present.

## Verification

Run from the project root on Fedora:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_temporal.py
.venv/bin/python tools/smoke_temporal.py --json
.venv/bin/python tools/smoke_replay.py
.venv/bin/python -m facepass --doctor
.venv/bin/python -m pip check
```

The temporal smoke helper covers landmark-only, photometric-only, aligned
cross-group, replay-conflict, missing-context and sampling-gap cases. Unit tests
cover persistence, bounded history, invalid timestamps, reset/expiry behavior,
session integration and strict scalar serialization; the complete suite currently
passes 193 tests. Physical webcam, replay-media,
lighting, user and GNOME preview evaluation remains outstanding because no camera is
available in the development environment.

## Assumptions and limits

The group partition is an engineering dependence model, not a learned statistical
independence estimate. A high-quality display or replay can reproduce several groups
and still receive weak temporal support. A real user with stillness, poor lighting,
glasses, motion blur or missing background texture can remain neutral or unavailable.
The adjusted support is deliberately not a probability and has no acceptance
threshold. Physical data, held-out users and attack media are required before any
fusion policy could be considered for an authentication product.

Development pauses after milestone 8. Milestone 9 must define and validate any
adaptive observation policy before this diagnostic project can be considered for
further security review.

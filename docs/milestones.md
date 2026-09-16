# Incremental implementation plan

Complete one major milestone, explain it, provide tests/limits, then pause for
direction. PAM, GDM, GNOME authentication and `/etc/pam.d` stay outside scope.

| Milestone | Scope and acceptance criteria | Status |
| --- | --- | --- |
| 1 | Local camera/landmarks, bounded diagnostic harness, quality and non-success failures | Implemented; hardware/preview testing outstanding |
| 2 | Stable one-face tracking; detect discontinuities and clear history/evidence; no identity/liveness claim | Implemented; hardware face-switch testing outstanding |
| 3 | Subtle head/landmark motion and planar/background motion comparison; still people need no challenge | Experimental implementation complete; physical motion/attack evaluation outstanding |
| 4 | Blink sequence state machine, noise/failure discrimination, no mandatory blink | Planned |
| 5 | Iris/gaze changes relative to head with explicit resolution/visibility reliability | Planned |
| 6 | Multi-region face shading/color vs ambient scene and temporal lighting response | Planned, experimental |
| 7 | Replay/print texture and frequency evidence with confounder controls | Planned, experimental |
| 8 | Temporal consistency and fusion retaining individual scores and dependence groups | Planned |
| 9 | Validated adaptive observation thresholds, hard timeout and optional challenge API | Planned |
| 10 | Attack evaluation across the full scenario matrix and held-out conditions | Planned |
| 11 | Profile/optimize CPU, detector cadence and bounded cancellation | Planned |
| 12 | Versioned local daemon interface after extensive evaluation; identity remains separate | Planned |

Milestone 1 assumptions: ordinary V4L2 RGB webcam, trusted local OS/device/model,
one nearby face, CPU inference, 640 × 480 target, experimental quality thresholds.
No reliable depth, occlusion or liveness is established yet. Milestone 2
establishes heuristic observation continuity and bounded history, not identity.
Milestone 3 adds approximate pose and two weak, correlated motion evidence producers;
it does not enable fusion, SPOOF or LIVE. The model
does not expose calibrated per-mesh confidence, and no camera or Tk preview could
be tested here. See README for commands and evaluation.md for manual cases.

Development is paused after milestone 3. See milestone-3-report.md; milestone 4
(blink detection) requires new direction.

Future milestones must continue returning non-success until the quality, tracking
and fusion policy have enough validated evidence. Implementing a blink or a motion
detector alone is never permission to enable LIVE. A challenge cannot replace
identity recognition or bypass fusion/quality policy.

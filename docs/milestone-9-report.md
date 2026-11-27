# Milestone 9 — bounded adaptive observation

Milestone 9 is implemented for the Fedora GNOME standalone harness. It adds a
capture-duration policy around the milestone 8 diagnostics. It does not add PAM,
GDM, GNOME, identity recognition, enrollment, challenge issuance, or an
authentication decision.

## Policy

The default policy requires a 1.5-second minimum observation window and uses the
requested duration as the base limit. A clean early completion is allowed only when
the track is stable, quality is acceptable, at least eight samples span 600 ms,
landmark dynamics and photometric scene groups both have sustained support, their
support overlaps and agrees, motion is not suppressed, the illumination behavior is
consistent, and replay/conflict evidence is absent. This can end capture sooner, but
it still returns an overall `UNCERTAIN` result with confidence zero.

When the base limit is reached with partial, conflict-free evidence, the policy can
request one bounded two-second extension. It never extends a conflicted, unstable or
poor-quality observation and never extends twice. After the effective limit it emits
`observation_timeout`. The `adaptive=False` constructor option disables extensions
and early completion while preserving the same hard base limit.

Challenge issuance is explicitly disabled in this milestone. The scalar
`challenge_state` is always `disabled`; no nonce, action, response, camera control or
user prompt is generated.

## Diagnostics and reset behavior

`ObservationDiagnostics` records status, elapsed time, minimum/base/effective limits,
extension count, quality/stable-frame coverage, readiness, conflict count, completion
state and the disabled challenge state. The worker validator checks every field,
keeps limits bounded, rejects inconsistent states and requires historical policy data
to be terminal and attached to a closed track.

The policy clears with detector evidence on face loss, quality failure, stale capture,
tracking discontinuity, cancellation, model failure and cleanup failure. A clean,
fresh, quality-stable finish may retain policy and detector values as historical
diagnostics; historical values cannot authorize another attempt.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/smoke_observation.py
.venv/bin/python tools/smoke_observation.py --json
.venv/bin/python -m facepass --doctor
.venv/bin/python -m pip check
```

The milestone tests cover early completion gates, minimum-window behavior, one
bounded extension, hard timeout, conflict/quality/tracking suppression, disabled
adaptive mode, strict IPC schema checks and terminal session behavior. The complete
suite now passes 203 tests. Physical webcam, user, replay-media, usability and GNOME
preview measurements remain outstanding because no camera or Tk is available in the
development environment.

## Limits

The policy thresholds are engineering bounds, not calibrated security or usability
limits. A real person may time out because lighting, motion, glasses or background
texture prevent diverse support. A replay may satisfy the same cross-group gates and
finish early. Extensions increase observation time only; they do not improve the
evidence or prove liveness. Physical completion/nonresponse rates and held-out attack
evaluation must precede any future authentication policy.

Development pauses after milestone 9. Milestone 10 must evaluate real users and
attack media before adaptive behavior is considered for security use.

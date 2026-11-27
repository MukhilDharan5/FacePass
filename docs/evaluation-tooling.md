# Attempt-level evaluation

The recorder makes milestone 10 reproducible. It does not supply a physical attack
study or calibrate a decision policy. All captures from this version remain
`UNCERTAIN` or `INSUFFICIENT_QUALITY`, with global confidence zero and capture exit
status 2. Recording a labelled attempt is a research operation.

## Collect attempts

Run from the installed environment and use a POSIX filesystem that enforces 0600
file permissions. The repository's shared media mount may not enforce them; `/tmp`
is suitable for temporary tests. Use a private persistent directory for a study.
Every attempt includes camera/model failures and timeouts. Do not discard these
outcomes or relabel them as explicit `SPOOF` decisions.

```bash
.venv/bin/python -m facepass.evaluation capture \
  --output /tmp/facepass-attempts.jsonl --device /dev/video0 --duration 5 \
  --label bona_fide --scenario still --split development \
  --participant p001 --room room001 --media none --device-code cam001

.venv/bin/python -m facepass.evaluation capture \
  --output /tmp/facepass-attempts.jsonl --device /dev/video0 --duration 5 \
  --label attack --scenario moved_print --split development \
  --participant p001 --room room001 --media print001 --device-code cam001

.venv/bin/python -m facepass.evaluation report /tmp/facepass-attempts.jsonl
```

Labels are `bona_fide` or `attack`; splits are `development` or `held_out`.
Scenario, participant, room, media and camera metadata use short anonymous codes.
Use `none` for bona-fide media. Keep the mapping from these codes outside the
repository. Codes are pseudonyms, not a guarantee of anonymity.

The collector uses the same validated duration, cadence, optional fixed window and
one-extension contract as the regular harness. Model selection remains the local
pinned asset. Images, landmarks, appearance features and raw signal trajectories
are excluded from the attempt log. The log records which signal names became
available during the attempt and whether any update had acceptable quality and a
`STABLE` track. This explicitly defines a *usable capture* for reporting; it does
not mean a face was live or recognized.

The schema includes a UUID attempt ID, UTC timestamp, labels/codes, engine version,
policy ID, expected pinned model digest, capture configuration, source, enabled
policy flag, final decision/reason, elapsed milliseconds, frame count, usable
capture flag and observed signal names. The digest describes the configured asset;
a failed initialization is not proof that the model was loaded. Files are JSONL
with one complete newline-terminated record per attempt.

Version 0.13.0 writes attempt schema 2 and policy ID `diagnostic_v2`, including the
explicit `quality_profile` (`tolerant` by default; `--quality-profile strict` selects
the original gates). Schema-1 records with the original capture configuration
remain readable. Reports keep the old and new policies/configurations separate.

Unknown fields, biometric vectors, duplicate IDs/JSON keys, nonfinite numbers,
invalid types, partial lines and excessive sizes are refused. Writes require an
owned regular file with mode 0600 and one link, reject symlinks, hold an exclusive
file lock, validate the existing log and flush to disk. The reader accepts at most
100,000 records, each at most 16 KiB. The filesystem and same-user processes remain
trusted; logs are not signed evidence. Delete retained logs when no longer needed.

## Study design and interpretation

Use the scenario matrix in [evaluation.md](evaluation.md). Include still users,
glasses/occlusion, facial hair, skin-tone and exposure variation, indoor/outdoor
lighting, low texture, camera motion, print/display/replay media, moved media,
brightness-adapted replays, multiple faces, camera loss and model/cancellation
failures. Record independent sessions across people, rooms, cameras and media.
Predefine a study protocol and keep calibration data separate from held-out data.

The reporter rejects participant, room or non-`none` media codes that overlap
between development and held-out attempts within each source. Use genuinely new
conditions for held-out testing; changing a code without changing the condition
does not create a valid holdout. Camera generalization must be assessed separately.
An all-held-out file alone cannot prove separation from an unseen calibration file;
include both splits when auditing leakage.

Reports separate source (`physical` versus `synthetic`), engine/model/policy,
capture configuration, enabled policy flag and split into cohorts, then stratify
by scenario. Mixed cohorts are not merged into a single security score. External
records from a future enabled policy can use the same explicit schema with their
own version and policy IDs; enabling that flag in data does not enable this engine.

| Report field | Numerator | Denominator |
| --- | --- | --- |
| attack_acceptance | Attack attempts returning LIVE | All attack attempts, including failures/nonresponses |
| usable_attack_acceptance | Usable attack attempts returning LIVE | All usable attack attempts |
| bona_fide_spoof_rejection | Bona-fide attempts explicitly returning SPOOF | All bona-fide attempts |
| bona_fide_nonacceptance | Bona-fide attempts returning anything other than LIVE | All bona-fide attempts |
| attack_nonresponse / bona_fide_nonresponse | UNCERTAIN or INSUFFICIENT_QUALITY attempts in that label | All attempts in that label |
| usable_capture | Attempts with any acceptable STABLE update | All attempts in the cohort |
| signal_availability | Attempts in which that signal was available | All attempts in the cohort |

Empty denominators yield `null`, never an invented zero. Each rate includes counts
and a 95% Wilson interval. These intervals assume independent attempts; repeated
users/rooms/media may require a clustered analysis outside this tool. Latency
includes engine observation time; initialization is measured separately by capture
profiling. Reports include decision/reason counts and p50/p95/p99/max latency.

The report always sets `pad_performance_established: false`. For this disabled
policy, zero accepted attacks is expected by construction and bona-fide
nonacceptance is 100%. Neither measures discrimination. Synthetic records remain
explicitly separate and cannot replace physical evaluation. Do not present these
counts as calibrated APCER/BPCER or production authentication evidence.

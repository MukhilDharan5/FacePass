# Capture quality profiles — 0.13.0

The CLI, evaluation collector, local daemon and client use `tolerant` by default.
Use `--quality-profile strict` on capture commands to reproduce the earlier gates.
These profiles measure experimental capture usability, not identity or liveness.
Global confidence remains zero and capture always exits 2.

| Gate | Tolerant default | Strict comparison |
| --- | --- | --- |
| Face box width | At least 96 px | At least 120 px |
| Face box area / image area | At least 0.015 | At least 0.025 |
| Laplacian variance | At least 8 after bounded brightness normalization | At least 45 on raw pixels |
| Face mean grayscale luminance | 12–245 | 40–230 |
| Fraction of pixels ≤10 or ≥245 | At most 0.65 on either side | At most 0.40 on either side |
| 95th–5th percentile contrast span | At least 3 raw and 8 after brightness normalization | No additional contrast gate |
| Frame clipping, landmark validity and rough orientation | Original checks | Original checks |

Measurements use the face mesh hull rather than background pixels. Sharpness
excludes the hull boundary. For the tolerant check, brightness gain is
`min(4, max(1, 128 / max(mean_luminance, 16)))`. Evaluated sharpness is raw
Laplacian variance times gain squared; contrast is evaluated with gain. The
raw contrast floor and clipping/brightness limits still exclude lost detail.
Brightness normalization is a heuristic and does not classify skin tone.

The result publishes `profile`, raw `blur_variance`, `evaluated_blur_variance`,
raw `contrast_span`, luminance/clipping measurements, `warnings`, hard-failure
`reasons` and `landmark_enhanced`. Moderate blur (raw variance below 45), low
luminance (below 40), limited contrast (below 20) and small faces (below the strict
size limits) are warnings when their corresponding tolerant gate passes. Warnings
do not clear a track. A hard quality failure still clears detector history.

## Low-light landmark input

The tolerant profile samples at most 160 × 120 pixels from the full frame. When
the median grayscale value is 8–65 (upper bound excluded) and percentile contrast
is at least 3, a gamma lookup table lifts the scene toward luminance 85. Gamma is
bounded to 0.55–1 and pixel gain to threefold. Black stays black, white stays white,
and the lookup table is monotonic. Uniform/near-black or normally lit input is
left unchanged. Strict capture never enhances landmark input.

Only landmark inference uses this temporary enhanced full-frame copy. There is
one model pass, no face crop, no second inference, no sharpening and no extra disk
storage. All other image measurements and previews use the original frame.
Detection still checks up to four faces; multiple detections terminate capture.
The existing frame, history, deadline, worker and IPC limits remain in force.

## Records and protocol

Evaluation attempt schema 2 records the explicit `quality_profile`, with policy
ID `diagnostic_v2`. Schema-1 records with the original five capture fields remain
readable and retain their original configuration; they describe strict capture.
Reports keep versions, configurations and policies in separate cohorts.

Daemon wire protocol 2 includes the added quality fields and optional
`quality_profile` capture option. Upgrade the client and daemon together. Version-1
requests are refused explicitly rather than returning an unfamiliar result schema.

## Validation limits

Regression fixtures cover moderate blur, simultaneous dim/blurred detail, dim
faces against bright backgrounds, lower resolution/JPEG compression, smaller
faces, severe blur, flat/black/white capture, enhancement gain and monotonicity,
raw-pixel analysis, track continuity, history clearing and multiple-face refusal.
These are generated textures and geometry, not a physical camera or skin-tone
study. The profiles require diverse physical evaluation before calibration or
authentication can be considered.

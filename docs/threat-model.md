# RGB presentation-attack threat model

Current security status: **capture, continuity and experimental motion evidence only**. Every attempt is non-success.
There is no demonstrated PAD discrimination and no supported face unlock behavior.

Milestone 2 clears temporal history on loss, quality failure, staleness and detected
discontinuity, and refuses strong visible continuity changes. It cannot guarantee
that two compatible meshes/appearance features belong to the same person. A
similar-looking substitution, gradual change or coherent replay can pass tracking.
A stable track number is not an identity or a PAD result.

Milestone 3 attempts to avoid rewarding whole-scene movement and landmark motion
explained by a moving plane. Approximate pose and held-out planar residuals can
contribute only weak, correlated scores. They are not reliable print/replay
classifiers: a face model can hallucinate changing geometry on a moving photo,
and prerecorded video already contains genuine head and facial movement. Missing
motion or a planar explanation remains neutral rather than automatic SPOOF.
Background depth parallax, moving objects, camera rolling shutter, reflections,
autofocus and image interpolation can confound environment compensation. No
physical attack-resistance claim has been demonstrated.

The eventual component targets a person presenting a physical medium to an ordinary
RGB laptop webcam. The attacker may possess photos or videos of an enrolled person,
adjust replay brightness and move/tilt the medium. Camera, kernel, model assets and
the eventual daemon are assumed trustworthy; that assumption needs separate review
before authentication integration.

| Attack | Future evidence worth evaluating | Limit |
| --- | --- | --- |
| Static printed photo | Missing temporal evidence, print texture, planar shading | Still live people may also supply little evidence |
| Printed photo moved naturally | Planar transform fit, background-relative motion, shading/pose inconsistency | Small movements may reveal no reliable difference |
| Static photo on a phone/tablet/laptop | Display frequency/banding, glare, ambient response | High-quality displays may lack visible artifacts |
| Prerecorded face video | Replay-medium evidence and scene/face temporal inconsistencies | Video contains genuine blinks, gaze and facial motion |
| Display moved/tilted | Display-plane behavior vs internal face geometry and lighting | Perspective can imitate natural head movement |
| Brightness-adjusted replay | Directional shading, color and temporal ambient response | Attacker can adapt brightness to the room |
| Replay recorded in similar lighting | Cross-signal consistency and display evidence | Lighting may be fully consistent |
| Multiple people | Refuse whenever multiple credible faces are detected | A missed, distant or occluded second face is not detected |

Software-only RGB PAD cannot reliably prevent every high-quality replay, adaptive
display, 3D mask, prosthetic/partial mask, generated real-time video or manipulated
camera stream. An attacker controlling the camera driver, a virtual camera, the
daemon process, model code or host can bypass visual assumptions. Restricting the
harness input to `/dev/videoN` does not attest that a kernel device is a physical
camera. A remote operator or malware injecting coherent frames needs defenses at
the trusted capture/host boundary, not another blink heuristic.

Landmark z is a monocular model prediction, not depth measurement. Natural motion
can appear in prerecorded video; absence of blink is not proof of a spoof. Prints
and displays need not have moire or halftoning. Illumination varies with reflectance,
auto-exposure and scene geometry. These signals can supply only probabilistic,
fallible evidence and require evaluation together.

This must never be described as equivalent to Windows Hello IR/depth authentication.
Password fallback belongs to the future higher-level authentication system; it is
not implemented here. No system login configuration may be changed during these
milestones.

# RGB presentation-attack threat model

Current security status: **capture, continuity, experimental motion/blink/gaze/illumination/replay-artifact evidence, and bounded temporal/fusion diagnostics only**. Every attempt is non-success.
There is no demonstrated PAD discrimination and no supported face unlock behavior.

Version 0.12.0 adds evaluation/profiling tools and a same-user local research
service. These do not change the decision policy. The socket requires owned private
POSIX paths and Linux peer credentials, rejects arbitrary inputs, bounds concurrent
requests and preserves finite-process supervision. Disconnects and cancellation
stop capture; no identity, enrollment or images are exposed. The host and other
processes of the same UID remain trusted. Logs/metadata are unsigned research input
and cannot attest ground truth; synthetic reports and zero acceptance under a
disabled policy do not establish attack resistance. See
[daemon-protocol.md](daemon-protocol.md) and [evaluation-tooling.md](evaluation-tooling.md).

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

Milestone 4 attempts to avoid crediting an isolated collapsed eye sample, asymmetric
eyes, implausible geometry, sustained closure or a sequence spanning missing
observations. A complete bilateral closure/reopening earns only weak evidence;
no blink stays neutral. These checks reject some generated noise patterns, but
cannot determine whether the eye movement occurred in front of the camera now.
A prerecorded video or generated landmark-compatible animation can earn the same
score. Model hallucinations on a print/display can also imitate a sequence. The
offline identical-pattern example deliberately shows this limit; no physical
blink/replay accuracy has been measured. Uncalibrated geometry gates do not attest
eye visibility or prevent occlusion-based attacks.

Milestone 5 attempts to avoid rewarding whole-image movement, one-eye/noisy iris
changes and movement accompanied by substantial head rotation. Eye-local coordinates,
bilateral agreement, robust sampled displacement and approximate-pose gating provide
only a weak relative-iris score. They do not prove live eye orientation, calibrated
gaze direction, visible eyes or independent 3D movement. Reflections/occlusion and
landmark hallucinations can produce plausible contours. A prerecorded or generated
sequence can satisfy every gaze gate; its identical-pattern fixture receives the
same score. Blink, gaze and motion depend on the same model and cannot supply
independent security votes. There is no demonstrated physical gaze attack resistance.

Milestone 6 compares temporal spatial RGB response in seven face patches against
four outside-face tiles. Common channel gains are removed as possible exposure/WB
effects; uniform brightness or color changes, static asymmetric lighting and
different face/background reflectance stay neutral. Ambient/facial response
mismatch is an unvalidated diagnostic, never automatic SPOOF. Missing regions,
unusable pose, clipping/darkness or radiometric jumps clear illumination evidence;
head/scene motion and missing background-flow context suppress positive credit.

This does not reconstruct physical light direction, color temperature, surface
normals or depth. Fixed patches may contain facial hair, glasses or false landmark
geometry; background tiles may contain moving objects, colored walls or display
light. Local exposure/tone mapping, specularities and mixed lighting need not obey
the common-gain model. A real face may fail the gates, a print can respond to room
light, and replay/generated video can match spatial/RGB response. The identical
generated replay receives the same weak score. Regional fit residual is not a
validated flat-media discriminator. Illumination also depends on the same landmarks,
pose and background tracking; agreement among these producers is correlated.
No physical illumination/attack performance has been measured.

Milestone 7 measures seven camera/content-dependent artifact families in five face
patches, relative to four outside-face controls. It requires multifamily, multiregion
and temporal persistence before lowering the liveness-oriented signal to weak
suspicion. Global sensor/lighting patterns, one local glare patch and transient
patterns stay neutral in controlled fixtures. Missing artifacts never count as
live evidence, and this producer cannot directly return SPOOF.

This does not locate a physical print/display edge or establish the source of any
pixel pattern. Skin, facial hair, glasses, fabrics, architecture, LED lighting,
demosaicing, rolling shutter, denoising, compression, sharpening and resampling can
create similar spectra. Background controls may sample different content or miss
a local camera artifact. Motion blur can remove real attack artifacts; high-quality
displays/prints can expose none. A generated pattern explicitly receives the same
suspicious score when labeled as bona fide, demonstrating an unavoidable false-positive
case for these calculations. The family scores share pixels/math and are correlated.
No physical attack detection or bona-fide false-positive rate has been measured.

Milestone 8 makes temporal dependence explicit rather than adding an unvalidated
weighted sum. Head motion, facial geometry, blink and gaze are one landmark-dynamics
group; illumination is a photometric-scene group; replay artifacts are a suspicion
group. A group can be missing, suppressed by scene motion or in conflict with another
group. Two aligned positive groups may produce only weak temporal diagnostic support;
replay overlap or response mismatch is recorded as a conflict. The summary retains
individual scores, uses a bounded 64-sample/two-second scalar window and applies a
transparent dependence penalty. It cannot identify a physical live subject, and its
candidate flags for LIVE and SPOOF are disabled. A sophisticated replay can reproduce
multiple groups; a real user can fail one or more groups. No calibration or physical
cross-signal attack rate exists yet.

Milestone 9 adds duration control around those diagnostics. A stable,
quality-acceptable window with conflict-free two-group support may finish after the
minimum observation period, while partial conflict-free evidence can receive one
bounded extension. A conflict, unstable track or poor quality receives no extension
and reaches a hard timeout. Adaptive timing does not attest that a face is live: a
replay can satisfy the same readiness gates, and a bona-fide user can time out.
Challenge issuance remains disabled, and every completion remains `UNCERTAIN` with
confidence zero. Physical completion, nonresponse and held-out attack rates are not
measured.

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
The 0.14.0 experimental system component implements password fallback separately
from these research milestones. It requires a root-owned SFace enrollment, a
fresh randomized active challenge, root-peer verification, request nonces and a
bounded supervised camera worker. It cannot attest a physical camera or prevent
all adaptive replays. The heuristic cosine threshold has not been calibrated for
this user, laptop or population. Root-only enrollment protects against an ordinary
local process replacing the account's templates; an administrator or compromised
host can bypass those protections. See [face-unlock.md](face-unlock.md).

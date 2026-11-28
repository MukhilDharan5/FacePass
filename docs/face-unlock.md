# Experimental Fedora KDE face unlock — 0.14.0

This implements the requested authentication integration independently of the
research daemon. The research CLI remains fail-closed. The system authenticator
can authenticate only an enrolled local account, after SFace identity matching
and a newly randomized blink/head-turn challenge on one continuous face track.

## Laptop setup

Open **FacePass Setup** from KDE's application menu, or run `facepass-setup`.
The prepared offline installer runs in Konsole and asks for the laptop's
administrator password through `sudo`. It installs the system component, then
opens the enrollment window as your ordinary desktop user. Select **Enroll and
verify**, complete the PolicyKit administrator dialog, face the camera and follow
the instructions. A second fresh challenge verifies the captured templates before
saving them. Closing the window cancels capture and prevents template replacement.

After successful enrollment:

1. Lock your screen and choose **FacePass** below the password field.
2. Face the camera, blink slowly as instructed, then turn slightly left/right
   and return to the center when asked. Order and blink counts change each time.
3. If verification fails, enter your password. A blocked camera/worker is killed
   within the bounded authentication attempt; the next password attempt remains
   available. Selecting FacePass deliberately submits an empty PAM response.

At Plasma Login/SDDM, submit an empty password to start the camera challenge;
the stock login-manager theme is preserved. The explicit FacePass button applies
to KDE's default desktop lock screen. Custom shell packages need their own QML
integration. A KDE package update may replace the modified component; the empty
password path still works while PAM integration remains present. The installer
refuses unfamiliar PAM/QML layouts instead of guessing.

Check readiness with `facepass-auth status`; `authentication_enabled` requires
the root broker, enrollment and the KDE PAM integration. `facepass-auth verify`
runs a real camera challenge without locking the screen. Removing enrollment is
an administrator operation:

```bash
sudo /usr/local/bin/facepass-auth remove --uid "$(id -u)"
```

## Implementation

- A root-owned offline wheel environment lives in `/opt/facepass-auth/current`.
  System runtime never imports the user's editable project, user-site packages
  or environment-supplied Python paths. Every bundled wheel is hash checked.
- Pinned local MediaPipe, YuNet and SFace models are verified before inference.
  There are no runtime network requests. SFace produces normalized 128-value
  identity templates. Matching requires support from three enrollment views,
  using a heuristic mean cosine threshold of 0.60 (0.70 during capture).
- Enrollment stores 8–12 templates in `/var/lib/facepass/users/UID.json`, mode
  0600, within root-owned 0700 directories. Captured images and landmarks are
  transient. Symlinks, hardlinks, malformed/oversized files, wrong UIDs and model
  mismatches are refused. Re-enrollment keeps the old templates until the new
  capture and independent verification both pass, then replaces them atomically.
- `facepass-auth.service` exposes `/run/facepass/auth.sock`. Kernel peer
  credentials bind requests to root or the requested account; the PAM module
  accepts only a root broker. Exact nonce-bound final replies are mandatory.
  Only `kde`, `kde-face`, `plasmalogin` and `sddm` are accepted, for UIDs ≥1000.
  There is one shared camera lock, eight connection slots, and five attempts per
  minute per enrolled account. Busy/unavailable/error paths never grant access.
- Every attempt runs in a new, isolated process group with camera cleanup,
  process termination on disconnect/shutdown, a 30-second capture budget,
  25-second active challenge and 45-second supervisor limit. PAM adds its own
  47-second response deadline. A successful challenge is consumed once.
- The native module preserves nonempty passwords in `PAM_AUTHTOK` and defers to
  the original password stack. A successful face match completes only the PAM
  authentication phase; existing account and session checks remain. Empty face
  failures terminate that attempt rather than submitting an empty password to
  Fedora's password modules. An absent broker or absent enrollment takes the
  ordinary password path. Generated `password-auth`/`system-auth` are unchanged.
- Quality defaults tolerate moderate blur, dim detail and smaller resolved faces.
  Bounded low-light enhancement helps landmark and identity inference. Quality,
  tracking and preview use original pixels. Severe blur, near-black capture,
  lost geometry, multiple faces, identity/track changes and stale frames fail.
  See [capture-quality.md](capture-quality.md).

The installer records original files privately, installs the service/module,
verifies root-peer IPC before editing PAM, and rolls back failures. The system
service is hardened with a read-only filesystem, private temporary directory,
restricted device access, no added capabilities, resource limits and no network
address families beyond Unix sockets. SELinux labels are restored; the installer
does not disable SELinux. The laptop reports SELinux enforcing after its restart;
service startup and root-peer IPC are checked before PAM is changed.

## Limits and recovery

**This is experimental RGB authentication.** A randomized active challenge helps
against a fixed photo or an unresponsive recording, but does not establish robust
presentation-attack detection. It does not measure depth or use infrared. Adaptive
replay, manipulated camera input, realistic masks and a compromised host remain
material limitations. The identity threshold and challenge timing are engineering
defaults; false acceptance/rejection rates and population fairness have not been
measured. Moderate low-light/blur handling has synthetic usability coverage, not
a guarantee for every camera, environment or skin tone. Face login does not supply
your password to unlock a password-protected wallet or encrypted boot volume.

Verify ordinary password unlock and real face unlock on the laptop after setup.
Admin credentials, physical enrollment and these interactive checks cannot be
completed by claiming that automated tests saw your face.

To undo system integration:

```bash
sudo /usr/bin/python3 -I /opt/facepass-auth/install_auth.py --uninstall
```

Rollback restores original PAM/QML files if untouched, otherwise removes only
the FacePass marker blocks and preserves subsequent changes. It stops/removes the
service and module. Private enrollment templates remain until explicitly removed.
Password login continues to use Fedora's original configuration. Before enrolling,
an installed system has no face-authenticated accounts.

## Validation and sources

The automated suite exercises actual libpam with temporary configurations and
a compiled native module: face success/failure, password-token forwarding,
unavailable services, nonroot broker rejection, invalid nonces, malformed replies
and bounded stalls. Additional tests cover private/atomic enrollment, two fresh
capture challenges, cancellation, quality/identity/track loss, stale frames,
model integrity, single-camera serialization and worker cleanup. The Qt 6 test
loads the patched MainBlock with KDE's real component library and a minimal test
replacement for the greeter-registered enum, then checks FacePass visibility,
empty-response submission and unchanged ordinary-password submission. These
tests do not constitute biometric or physical attack calibration.

Primary implementation references:

- [OpenCV face detection and recognition documentation](https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html)
  and [official SFace implementation](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py).
- [KDE 6.7.5 greeter](https://invent.kde.org/plasma/kscreenlocker/-/tree/v6.7.5/greeter)
  and the installed KDE lock-screen components. Their existing license headers
  are preserved by the small reversible QML insertions.
- [Linux-PAM 1.7.2 public headers](https://github.com/linux-pam/linux-pam/tree/v1.7.2/libpam/include/security).
  Vendored unmodified headers and their copyright terms are under `native/include`
  and `native/Linux-PAM-Copyright`. The FacePass bridge is MIT licensed.
- Model license files are distributed beside the local models. YuNet SHA256:
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
  SFace SHA256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.

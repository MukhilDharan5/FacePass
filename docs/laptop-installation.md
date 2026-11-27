# Laptop installation — 11 October 2026

FacePass 0.13.0 is installed for `mukhildharan` on Fedora KDE Plasma. The installed
copy uses an isolated Python environment and a verified local model, independently
of the development checkout or its mounted project drive.

Open **FacePass** from KDE's application menu, or run in Konsole:

```bash
facepass --doctor
facepass --device /dev/video0 --preview
```

The menu launcher runs preview in Konsole and keeps the terminal open so the final
diagnostics remain visible. Close the preview or press Escape to cancel. Every
capture returns status 2; authentication remains disabled.

Installed locations:

- Application and environment: `/home/mukhildharan/.local/share/facepass`
- Commands: `/home/mukhildharan/.local/bin/facepass`, `facepass-evaluate`,
  `facepass-daemon`, `facepass-client`
- KDE launcher: `/home/mukhildharan/.local/share/applications/facepass.desktop`
- Installation receipt: `/home/mukhildharan/.local/share/facepass/install.json`
- Verification: `/home/mukhildharan/.local/share/facepass/verification.json`
- Quality verification: `/home/mukhildharan/.local/share/facepass/quality-verification.json`

The installed wrappers select the installed model through `FACEPASS_MODEL` while
preserving the terminal's working directory and relative output paths. Explicit
`--model` arguments still override that default.

Verification from outside the checkout passed for package/model integrity,
dependency consistency, all four command entry points, doctor, supervised capture,
evaluation recording/reporting, daemon/client startup and cleanup, and a hidden
generated-PNG Tk preview. The KDE launcher validates and its application cache was
refreshed. No service or autostart was enabled.

Outside the workspace sandbox, doctor finds accessible `/dev/video0` and
`/dev/video1` camera nodes on this laptop. Verification used a nonexistent device
to exercise capture failure and did not capture webcam images. A camera preview
can now be run from the installed application; physical attack evaluation and
calibrated authentication remain outstanding.

The 0.12.1 diagnostic patch corrects cleared-history labels and reports the final
quality measurements and blur guidance. All 250 regression tests passed. The
standalone installation was reverified, including the daemon/client and hidden
preview. A synthetic good-frame → blur → timeout reproduction confirmed the
installed terminal now reports `quality_not_acceptable` instead of claiming the
detectors are unimplemented. That patch left quality thresholds and authentication
policy unchanged.

The 0.13.0 update enables the tolerant capture profile by default, including for
the existing KDE launcher. Moderate blur, smaller resolved faces and dim facial
detail can continue tracking with warnings. Landmark detection receives a bounded
gamma adjustment in dim scenes; image-based analysis retains original pixels.
Use `facepass --preview --quality-profile strict` to compare the original gates.
See [capture quality profiles](capture-quality.md) for thresholds and limits.

All 273 tests and all 12 project checks passed. The upgraded standalone copy was
verified from `/tmp`, including both CLI quality profiles, evaluation schema 2,
daemon/client wire version 2 with profile forwarding, hidden XWayland preview,
KDE launcher validation/cache refresh and dependency/model integrity. Generated
dim/blurred frames reached STABLE without track resets in the installed package;
severe blur, black and white capture still failed. These generated checks do not
establish physical camera, skin-tone or presentation-attack performance.

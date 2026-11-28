# Version 2 local research protocol

The service runs in the foreground as the ordinary desktop user. It provides
observation diagnostics; authentication is disabled. It installs no service,
authentication hooks, recognition/enrollment data, desktop extension or GDM/SDDM changes.
It does not retain diagnostic images or accept arbitrary model/video paths from
clients. Physical evaluation must precede any use in an authentication component.

```bash
.venv/bin/python -m facepass.daemon --model models/face_landmarker.task
.venv/bin/python -m facepass.client status
.venv/bin/python -m facepass.client observe --request-id trial001 --duration 5
.venv/bin/python -m facepass.client cancel trial001
```

The default socket is `$XDG_RUNTIME_DIR/facepass/daemon.sock`, or
`/tmp/facepass-UID/daemon.sock` when no runtime directory is configured. `--socket`
selects a custom path on both programs. Its immediate parent must be owned by the
current user with mode 0700 and have no symlink parents. The socket has mode 0600.
The service refuses an existing path; it does not automatically delete a live or
stale socket. After checking that its previous owner has stopped, a user can remove
a stale socket. Filesystems that cannot enforce private POSIX modes are refused.

Linux `SO_PEERCRED` requires the same UID on both sides. Other local processes
running as that user and the host are trusted. There is no TCP listener or remote
access, and no camera/model inference while idle. At most eight connections and
one supervised capture worker exist. An additional observation receives `busy`.
Different research daemons/harnesses can still compete for the hardware camera;
there is no system-wide capture lock.

## Framing and requests

Connect an AF_UNIX stream socket. Send exactly one UTF-8 JSON object followed by
one newline. Keep an observation connection open until its final response; do not
half-close the writing side. Responses are newline-delimited JSON. Close the
connection after status/cancel/error/final. Extra request bytes cancel an active
observation, and multiple commands on one connection are refused.

Requests are at most 4096 bytes. A partial request has a three-second receive
timeout. Duplicate JSON keys, NaN/Infinity, unsupported versions, unknown fields,
invalid identifiers/types/options and unknown operations are rejected. Request IDs
use 1–64 ASCII letters, digits, underscore or hyphen. Use a fresh ID per attempt.

Observe:

```json
{"version":2,"operation":"observe","request_id":"trial001","options":{"device":"/dev/video0","duration":5.0,"fps":12,"fixed_window":false,"extension_duration":2.0,"quality_profile":"tolerant"}}
```

All six option keys may be omitted and receive the defaults above. Duration is
0.5–30 seconds, fps is an integer in 5–30, the device is a local `/dev/videoN` node,
fixed_window is boolean, extension_duration is 0–5 seconds, and quality_profile is
`tolerant` or `strict`. Models are selected
at service startup and verified using the existing pinned asset policy. Preview,
frame retention, file/URL/virtual stream arguments and identity lookup are absent.

Status and cancellation use independent connections:

```json
{"version":2,"operation":"status","request_id":"status001"}
{"version":2,"operation":"cancel","request_id":"cancel001","target_id":"trial001"}
```

`target_id` must match the currently active observation. An unknown/completed target
returns `unknown_attempt`. The cancel acknowledgement means the cancellation flag
was set. The original observation stream supplies the terminal diagnostic result
after worker cleanup. The Python client can also be used programmatically:

```python
from facepass.daemon import stream_request
for message in stream_request("/tmp/facepass-1000/daemon.sock", {
    "version": 2, "operation": "observe", "request_id": "trial001", "options": {},
}):
    print(message["kind"])
```

## Responses and cleanup

Every response has `version`, the originating `request_id`, and `kind`.

| kind | Additional fields | Meaning |
| --- | --- | --- |
| accepted | authentication_enabled=false | A capture slot is reserved; model/camera initialization may still fail |
| update | result | Known, finite scalar diagnostics from the existing worker schema |
| final | result | Validated terminal result, after the finite worker exits and is reaped |
| status | engine_version, authentication_enabled=false, active_request_id | Idle has null active_request_id |
| cancel_requested | none | Matching active attempt has been asked to cancel |
| error | error | Short error code; malformed requests may have null request_id |

Current results permit only UNCERTAIN/INSUFFICIENT_QUALITY at confidence 0, with
disabled fusion candidates and disabled challenge state. Frames, geometry vectors,
embeddings and previews never cross this socket. Updates are transient; a clean
finish can retain only the existing historical scalar diagnostics, whose live
signal availability is false and reliability is zero. Cancellation/failure clears
detector state. The client independently validates the result schema and refuses
an unexpected success, wrong peer UID, oversized/truncated response or early EOF.

The supervisor enforces a 20-second initialization timeout, a two-second frame
stall timeout, the configured base plus one allowed extension plus 500 ms of
completion grace, and bounded worker shutdown. Process termination gets 250 ms
before a kill, then at most one second to reap. These are supervision bounds, not
measured physical camera latency. No cadence optimization skips full-frame face
detection or changes evidence thresholds.

A client disconnect, additional stream input, explicit cancel, Ctrl+C or SIGTERM
sets cancellation. The supervisor cleans up a blocked native worker before the
capture slot is released. A slow or broken reader cannot indefinitely keep camera
memory: response writes have a 500-ms timeout. Shutdown waits for handler cleanup
and removes only the socket inode the server created. The client observe command
always exits 2; status/cancel tooling may exit 0. These statuses are never login
authorization. Wire version 2 accompanies package version 0.13.0's expanded strict
quality result schema (profile, warnings, contrast, evaluated sharpness and
landmark enhancement state). Upgrade client and daemon together; version-1
requests receive `unsupported_protocol`. Any incompatible change requires a new
protocol version.

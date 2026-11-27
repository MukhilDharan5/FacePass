"""CLI for the versioned same-user research daemon."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from .daemon import PROTOCOL_VERSION, default_socket, stream_request
from .runtime import add_capture_arguments, options_from_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket())
    sub = parser.add_subparsers(dest="operation", required=True)
    observe = sub.add_parser("observe")
    add_capture_arguments(observe)
    observe.add_argument("--request-id", default=None)
    sub.add_parser("status")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("target_id")
    args = parser.parse_args(argv)
    request_id = getattr(args, "request_id", None) or uuid.uuid4().hex
    request = {"version": PROTOCOL_VERSION, "request_id": request_id, "operation": args.operation}
    try:
        if args.operation == "observe":
            request["options"] = asdict(options_from_args(args))
        elif args.operation == "cancel":
            request["target_id"] = args.target_id
        last = None
        for message in stream_request(args.socket, request):
            print(json.dumps(message, allow_nan=False), flush=True)
            last = message
        if last is None or last["kind"] == "error" or args.operation == "observe":
            return 2
        return 0
    except KeyboardInterrupt:
        if args.operation == "observe":
            try:
                cancellation = {"version": PROTOCOL_VERSION, "request_id": uuid.uuid4().hex,
                                "operation": "cancel", "target_id": request_id}
                for message in stream_request(args.socket, cancellation, timeout=2):
                    print(json.dumps(message), flush=True)
            except (OSError, ValueError):
                pass
        return 2
    except (OSError, ValueError) as error:
        parser.exit(2, f"client error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

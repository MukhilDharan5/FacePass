"""Attempt-level evaluation records and descriptive reports; no calibration claims."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from . import __version__
from .model_assets import MODEL_SHA256, default_model_path
from .performance import distribution
from .results import SIGNAL_NAMES
from .runtime import MAX_CAPTURE_FRAMES, CaptureOptions, add_capture_arguments, options_from_args
from .supervisor import supervise


MAX_RECORD_BYTES = 16_384
MAX_RECORDS = 100_000
FIELDS = {"schema_version", "attempt_id", "timestamp_utc", "label", "scenario", "split",
          "participant", "room", "media", "device_code", "source", "engine_version",
          "policy_id", "model_sha256", "config", "authentication_enabled", "decision",
          "reason", "elapsed_ms", "frame_count", "usable_captured", "observed_signals"}


def code(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", value) is not None


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def validate_record(record):
    if not isinstance(record, dict) or set(record) != FIELDS or type(record["schema_version"]) is not int or record["schema_version"] not in (1, 2):
        raise ValueError("unknown evaluation schema")
    for field in ("attempt_id", "scenario", "participant", "room", "media", "device_code", "engine_version", "policy_id", "reason"):
        if not code(record[field]):
            raise ValueError(f"invalid {field}; use a short non-identifying code")
    try:
        timestamp = datetime.fromisoformat(record["timestamp_utc"])
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
    except (TypeError, AttributeError, ValueError) as error:
        raise ValueError("invalid UTC timestamp") from error
    if record["label"] not in ("bona_fide", "attack") or record["split"] not in ("development", "held_out"):
        raise ValueError("invalid label or split")
    if record["source"] not in ("physical", "synthetic"):
        raise ValueError("source must be physical or synthetic")
    if record["decision"] not in ("LIVE", "SPOOF", "UNCERTAIN", "INSUFFICIENT_QUALITY"):
        raise ValueError("invalid decision")
    for field in ("authentication_enabled", "usable_captured"):
        if type(record[field]) is not bool:
            raise ValueError(f"invalid {field}")
    if not record["authentication_enabled"] and record["decision"] in ("LIVE", "SPOOF"):
        raise ValueError("a disabled policy cannot classify LIVE or SPOOF")
    if not isinstance(record["model_sha256"], str) or re.fullmatch(r"[a-f0-9]{64}", record["model_sha256"]) is None:
        raise ValueError("invalid model digest")
    for field, limit in (("elapsed_ms", 60_000), ("frame_count", MAX_CAPTURE_FRAMES)):
        if type(record[field]) is not int or not 0 <= record[field] <= limit:
            raise ValueError(f"invalid {field}")
    config_fields = set(CaptureOptions.__dataclass_fields__)
    if record["schema_version"] == 1:
        config_fields.remove("quality_profile")
    if not isinstance(record["config"], dict) or set(record["config"]) != config_fields:
        raise ValueError("invalid capture config")
    CaptureOptions(**({**record["config"], "quality_profile": "strict"}
                      if record["schema_version"] == 1 else record["config"]))
    signals = record["observed_signals"]
    if not isinstance(signals, list) or any(not isinstance(name, str) or name not in SIGNAL_NAMES for name in signals) or len(signals) != len(set(signals)):
        raise ValueError("invalid signal summary")
    return record


class AttemptRecorder:
    def __init__(self):
        self.usable = False
        self.signals = set()

    def update(self, result, preview=None):
        self.usable |= result["quality"]["acceptable"] and result["tracking"]["state"] == "STABLE"
        self.signals.update(name for name, signal in result["signals"].items() if signal["available"])

    def record(self, result, config, *, label, scenario, split, participant, room, media, device_code,
               source="physical", attempt_id=None):
        return validate_record({
            "schema_version": 2, "attempt_id": attempt_id or uuid.uuid4().hex,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "label": label,
            "scenario": scenario, "split": split, "participant": participant, "room": room,
            "media": media, "device_code": device_code, "source": source,
            "engine_version": __version__, "policy_id": "diagnostic_v2", "model_sha256": MODEL_SHA256,
            "config": asdict(config), "authentication_enabled": False, "decision": result["decision"],
            "reason": result["reason"], "elapsed_ms": result["observation_duration_ms"],
            "frame_count": result["frame_count"], "usable_captured": self.usable,
            "observed_signals": sorted(self.signals),
        })


def read_records(stream):
    records, seen = [], set()
    while True:
        line = stream.readline(MAX_RECORD_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
            raise ValueError("oversized or incomplete evaluation record")
        record = validate_record(strict_json(line))
        if record["attempt_id"] in seen:
            raise ValueError("duplicate attempt identifier")
        seen.add(record["attempt_id"])
        records.append(record)
        if len(records) > MAX_RECORDS:
            raise ValueError("too many evaluation records")
    return records


def append_record(path, record):
    """Serialize under a file lock; refuse symlinks, special files and shared modes."""
    validate_record(record)
    encoded = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("oversized record")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "r+b") as output:
        info = os.fstat(output.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            raise ValueError("evaluation log must be a private 0600 regular file on a POSIX filesystem")
        fcntl.flock(output, fcntl.LOCK_EX)
        records = read_records(output)
        if len(records) >= MAX_RECORDS or any(item["attempt_id"] == record["attempt_id"] for item in records):
            raise ValueError("duplicate attempt or full evaluation log")
        output.seek(0, os.SEEK_END)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def rate(numerator, denominator):
    if not denominator:
        return {"numerator": numerator, "denominator": 0, "rate": None, "wilson_95": None}
    z = 1.959963984540054
    p = numerator / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator * denominator)) / scale
    return {"numerator": numerator, "denominator": denominator, "rate": p,
            "wilson_95": [max(0, center - radius), min(1, center + radius)]}


def summarize(records):
    attacks = [item for item in records if item["label"] == "attack"]
    bona_fide = [item for item in records if item["label"] == "bona_fide"]
    usable_attacks = [item for item in attacks if item["usable_captured"]]
    counts = {decision: sum(item["decision"] == decision for item in records)
              for decision in ("LIVE", "SPOOF", "UNCERTAIN", "INSUFFICIENT_QUALITY")}
    nonresponse = lambda items: sum(item["decision"] in ("UNCERTAIN", "INSUFFICIENT_QUALITY") for item in items)
    return {"attempt_count": len(records), "decision_counts": counts,
            "attack_acceptance": rate(sum(item["decision"] == "LIVE" for item in attacks), len(attacks)),
            "usable_attack_acceptance": rate(sum(item["decision"] == "LIVE" for item in usable_attacks), len(usable_attacks)),
            "bona_fide_spoof_rejection": rate(sum(item["decision"] == "SPOOF" for item in bona_fide), len(bona_fide)),
            "bona_fide_nonacceptance": rate(sum(item["decision"] != "LIVE" for item in bona_fide), len(bona_fide)),
            "attack_nonresponse": rate(nonresponse(attacks), len(attacks)),
            "bona_fide_nonresponse": rate(nonresponse(bona_fide), len(bona_fide)),
            "usable_capture": rate(sum(item["usable_captured"] for item in records), len(records)),
            "latency_ms": distribution([item["elapsed_ms"] for item in records]),
            "reason_counts": {reason: sum(item["reason"] == reason for item in records) for reason in sorted({item["reason"] for item in records})},
            "signal_availability": {name: rate(sum(name in item["observed_signals"] for item in records), len(records)) for name in SIGNAL_NAMES}}


def build_report(records):
    seen, development, held_out = set(), [], []
    cohorts = {}
    for item in records:
        validate_record(item)
        if item["attempt_id"] in seen:
            raise ValueError("duplicate attempt identifier")
        seen.add(item["attempt_id"])
        (development if item["split"] == "development" else held_out).append(item)
        keys = ("engine_version", "policy_id", "model_sha256", "config", "authentication_enabled", "source", "split")
        identity = {key: item[key] for key in keys}
        key = json.dumps(identity, sort_keys=True)
        cohorts.setdefault(key, (identity, []))[1].append(item)
    for source in ("physical", "synthetic"):
        for dimension in ("participant", "room", "media"):
            dev = {item[dimension] for item in development if item["source"] == source and (dimension != "media" or item["media"] != "none")}
            held = {item[dimension] for item in held_out if item["source"] == source and (dimension != "media" or item["media"] != "none")}
            if dev & held:
                raise ValueError(f"held-out leakage in {source} {dimension}: use disjoint codes")
    return {"schema_version": 1, "attempt_count": len(records),
            "physical_attempt_count": sum(item["source"] == "physical" for item in records),
            "pad_performance_established": False,
            "interpretation": "Descriptive attempt counts only. Disabled policies and synthetic fixtures cannot establish PAD performance. Nonresponses remain in all-attempt denominators; usable means at least one quality-acceptable STABLE update. Wilson intervals assume independent attempts; repeated users/media may be correlated.",
            "cohorts": [{"identity": identity, "summary": summarize(items),
                         "scenarios": {scenario: summarize([item for item in items if item["scenario"] == scenario]) for scenario in sorted({item["scenario"] for item in items})}}
                        for identity, items in (cohorts[key] for key in sorted(cohorts))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    capture = sub.add_parser("capture", help="Record one labelled physical webcam attempt as scalars")
    add_capture_arguments(capture)
    capture.add_argument("--model", default=default_model_path())
    capture.add_argument("--output", required=True, help="Private JSONL log, for example /tmp/facepass-attempts.jsonl")
    for name in ("label", "scenario", "split", "participant", "room", "media", "device-code"):
        capture.add_argument("--" + name, required=True)
    report = sub.add_parser("report", help="Validate JSONL attempts and emit a stratified JSON report")
    report.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "report":
            with args.input.open("rb") as source:
                result = build_report(read_records(source))
            print(json.dumps(result, allow_nan=False, indent=2))
            return 0
        config = options_from_args(args)
        metadata = {name: getattr(args, name) for name in ("label", "scenario", "split", "participant", "room", "media", "device_code")}
        recorder = AttemptRecorder()
        # Validate metadata before opening the camera.
        from .results import failure_result
        recorder.record(failure_result("metadata_check"), config, **metadata)
        result = supervise(config.worker_command(args.model), duration=config.duration,
                           extension_duration=config.maximum_extension, on_update=recorder.update)
        record = recorder.record(result, config, **metadata)
        append_record(args.output, record)
        print(json.dumps(record, allow_nan=False))
        return 2  # A recorded observation is never authentication success.
    except (OSError, ValueError) as error:
        parser.exit(2, f"evaluation error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

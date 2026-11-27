import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facepass.evaluation import AttemptRecorder, append_record, build_report, main, rate, read_records, strict_json, summarize, validate_record
from facepass.results import failure_result
from facepass.runtime import CaptureOptions


def record(attempt_id="a1", **changes):
    value = AttemptRecorder().record(failure_result("example"), CaptureOptions(),
        label="attack", scenario="display_replay", split="development", participant="p1",
        room="r1", media="m1", device_code="camera1", source="synthetic", attempt_id=attempt_id)
    value.update(changes)
    return value


class EvaluationTests(unittest.TestCase):
    def test_denominators_keep_failures_and_nonresponses_visible(self):
        records = [record("a1", authentication_enabled=True, decision="LIVE", usable_captured=True),
                   record("a2", decision="INSUFFICIENT_QUALITY"),
                   record("b1", label="bona_fide", authentication_enabled=True, decision="SPOOF"),
                   record("b2", label="bona_fide", decision="UNCERTAIN")]
        summary = summarize(records)
        self.assertEqual(summary["attack_acceptance"]["rate"], .5)
        self.assertEqual(summary["usable_attack_acceptance"]["rate"], 1)
        self.assertEqual(summary["bona_fide_spoof_rejection"]["rate"], .5)
        self.assertEqual(summary["bona_fide_nonacceptance"]["rate"], 1)
        self.assertEqual(summary["bona_fide_nonresponse"]["rate"], .5)

    def test_undefined_rates_and_small_sample_intervals(self):
        self.assertIsNone(rate(0, 0)["rate"])
        interval = rate(0, 2)["wilson_95"]
        self.assertAlmostEqual(interval[0], 0)
        self.assertGreater(interval[1], .5)
        self.assertEqual(build_report([])["attempt_count"], 0)

    def test_disabled_policy_and_synthetic_results_never_claim_pad_accuracy(self):
        report = build_report([record()])
        self.assertFalse(report["pad_performance_established"])
        self.assertEqual(report["physical_attempt_count"], 0)
        self.assertEqual(report["cohorts"][0]["summary"]["attack_nonresponse"]["rate"], 1)

    def test_source_policy_configuration_and_split_are_separate_cohorts(self):
        report = build_report([record(), record("a2", source="physical"), record("a3", policy_id="other")])
        self.assertEqual(len(report["cohorts"]), 3)
        self.assertEqual(report["physical_attempt_count"], 1)

    def test_legacy_records_stay_readable_and_new_profiles_are_separate_cohorts(self):
        legacy = record("legacy", schema_version=1, engine_version="0.12.1", policy_id="diagnostic_v1")
        legacy["config"].pop("quality_profile")
        before = copy.deepcopy(legacy)
        self.assertEqual(validate_record(legacy), before)
        strict = record("strict")
        strict["config"]["quality_profile"] = "strict"
        report = build_report([legacy, record("tolerant"), strict])
        self.assertEqual(report["attempt_count"], 3)
        self.assertEqual(len(report["cohorts"]), 3)
        self.assertEqual(legacy, before)

    def test_quality_profile_is_required_only_in_new_record_schema(self):
        value = record()
        self.assertEqual(value["schema_version"], 2)
        self.assertEqual(value["config"]["quality_profile"], "tolerant")
        value["config"].pop("quality_profile")
        with self.assertRaises(ValueError):
            validate_record(value)
        value = record(schema_version=1)
        with self.assertRaises(ValueError):
            validate_record(value)

    def test_held_out_codes_must_be_disjoint_in_each_dimension(self):
        held = record("held", split="held_out", participant="p2", room="r2", media="m2")
        self.assertEqual(build_report([record(), held])["attempt_count"], 2)
        for dimension in ("participant", "room", "media"):
            with self.subTest(dimension=dimension):
                leaking = copy.deepcopy(held)
                leaking[dimension] = record()[dimension]
                with self.assertRaisesRegex(ValueError, "leakage"):
                    build_report([record(), leaking])

    def test_bona_fide_media_none_does_not_count_as_leakage(self):
        first = record(media="none", label="bona_fide")
        held = record("held", split="held_out", participant="p2", room="r2", media="none", label="bona_fide")
        self.assertEqual(build_report([first, held])["attempt_count"], 2)

    def test_duplicate_ids_and_unknown_biometric_fields_are_refused(self):
        with self.assertRaises(ValueError):
            build_report([record(), record()])
        bad = record()
        bad["landmarks"] = [[.2, .3]]
        with self.assertRaises(ValueError):
            validate_record(bad)

    def test_schema_rejects_types_ranges_metadata_and_disabled_success(self):
        for changes in ({"schema_version": True}, {"elapsed_ms": True}, {"frame_count": 1101},
                        {"source": "webcam_maybe"}, {"timestamp_utc": "2026-10-11"},
                        {"participant": "a person@example.com"}, {"observed_signals": ["embedding"]},
                        {"decision": "LIVE"}, {"authentication_enabled": 0}, {"model_sha256": "invalid"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record(record(**changes))
        for options in ({"duration": float("nan")}, {"fps": True}, {"device": "https://camera"}, {"fixed_window": 1},
                        {"quality_profile": "always_accept"}):
            bad = record()
            bad["config"].update(options)
            with self.assertRaises(ValueError):
                validate_record(bad)

    def test_recorder_remembers_usable_updates_after_terminal_cleanup(self):
        recorder = AttemptRecorder()
        update = failure_result("observing")
        update["quality"]["acceptable"] = True
        update["tracking"]["state"] = "STABLE"
        update["signals"]["blink"]["available"] = True
        recorder.update(update)
        final = recorder.record(failure_result("finished"), CaptureOptions(), label="bona_fide",
                                scenario="still", split="development", participant="p1", room="r1", media="none", device_code="cam1")
        self.assertTrue(final["usable_captured"])
        self.assertEqual(final["observed_signals"], ["blink"])

    def test_private_log_append_roundtrip_and_duplicate_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            append_record(path, record())
            append_record(path, record("a2"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with path.open("rb") as source:
                self.assertEqual(len(read_records(source)), 2)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                append_record(path, record())
            self.assertEqual(path.read_bytes(), before)

    def test_shared_file_symlink_and_partial_log_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            path.write_bytes(b"partial")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                append_record(path, record())
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                append_record(path, record())
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                append_record(link, record())
            self.assertEqual(path.read_bytes(), b"partial")

    def test_json_nan_duplicates_incomplete_and_oversized_lines_fail(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(data)
        for data in (b"x" * 20_000 + b"\n", json.dumps(record()).encode(), b"{}\n"):
            with self.assertRaises(ValueError):
                read_records(io.BytesIO(data))

    def test_capture_and_report_cli_work_without_enabling_authentication(self):
        with tempfile.TemporaryDirectory() as directory, patch("facepass.evaluation.supervise", return_value=failure_result("camera_open_failed")), patch("sys.stdout", new_callable=io.StringIO):
            path = Path(directory) / "log.jsonl"
            status = main(["capture", "--output", str(path), "--label", "attack", "--scenario", "print",
                           "--split", "development", "--participant", "p1", "--room", "r1", "--media", "m1", "--device-code", "cam1"])
            self.assertEqual(status, 2)
            self.assertEqual(main(["report", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()

"""Run the complete offline engineering checks with concise, repeatable output."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--skip-model", action="store_true", help="Explicitly skip the verified real-model smoke check")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    checks = [("unit_and_integration_tests", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"]),
              ("dependency_consistency", ["-m", "pip", "check"])]
    if not args.skip_model:
        checks.append(("verified_model", ["tools/smoke_model.py", "--model", str(args.model.resolve())]))
        checks.append(("real_daemon_missing_camera", ["tools/smoke_daemon.py", "--model", str(args.model.resolve())]))
    for name in ("tracking", "motion", "blink", "gaze", "illumination", "replay", "temporal", "observation"):
        checks.append((name + "_scenarios", [f"tools/smoke_{name}.py"] + ([] if name == "tracking" else ["--json"])))
    results = []
    with tempfile.TemporaryDirectory(prefix="facepass-verify-") as cache:
        env = {**os.environ, "MPLCONFIGDIR": cache, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        for name, command in checks:
            before = time.monotonic()
            try:
                process = subprocess.run([sys.executable, *command], cwd=root, env=env,
                                         capture_output=True, text=True, timeout=300)
                result = {"check": name, "passed": process.returncode == 0,
                          "duration_ms": round((time.monotonic() - before) * 1000), "exit_code": process.returncode}
                if not result["passed"]:
                    result["failure_output"] = (process.stdout + process.stderr)[-6000:]
                elif name == "unit_and_integration_tests":
                    result["test_summary"] = process.stderr.strip().splitlines()[-4:]
            except subprocess.TimeoutExpired:
                result = {"check": name, "passed": False, "reason": "check_timeout"}
            results.append(result)
            print(json.dumps(result), flush=True)
    passed = all(result["passed"] for result in results)
    print(json.dumps({"all_checks_passed": passed, "check_count": len(results),
                      "model_check_skipped": args.skip_model, "physical_validation_performed": False}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

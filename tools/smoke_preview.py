"""Verify Tk display/image/update/cleanup without opening a camera or visible window."""

import argparse
import base64
import json
import sys

from facepass.cli import Preview
from facepass.desktop import preview_failure_hint, session_diagnostics
from facepass.results import failure_result


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    preview = None
    report = {**session_diagnostics(), "camera_opened": False,
              "physical_camera_validation_performed": False}
    try:
        preview = Preview()
        preview._root.withdraw()
        import cv2
        import numpy as np
        ok, encoded = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
        if not ok:
            raise RuntimeError("synthetic preview encoding failed")
        preview.update(failure_result("synthetic_preview_check"), base64.b64encode(encoded).decode("ascii"))
        if not preview.pump() or (preview._image.width(), preview._image.height()) != (16, 16):
            raise RuntimeError("synthetic preview update failed")
        backend = preview._root.tk.call("tk", "windowingsystem")
        preview.close()
        if preview.pump():
            raise RuntimeError("preview did not close")
        report.update(tk_display_connection_verified=True, windowing_system=backend,
                      synthetic_png_preview_verified=True, cleanup_verified=True)
        print(json.dumps(report))
        return 0
    except Exception as error:
        report.update(preview_smoke_passed=False, failure_type=type(error).__name__)
        print(json.dumps(report))
        print(preview_failure_hint(error), file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Session compatibility checks use environment fixtures, never a physical camera."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import patch

from facepass.cli import main
from facepass.desktop import preview_diagnostics, preview_failure_hint, session_diagnostics
from facepass.results import failure_result


KDE_WAYLAND = {"XDG_CURRENT_DESKTOP": "KDE", "XDG_SESSION_DESKTOP": "KDE",
               "XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0",
               "QT_QPA_PLATFORM": "wayland", "QT_QPA_PLATFORMTHEME": "kde"}


class DesktopTests(unittest.TestCase):
    def test_plasma_wayland_uses_xwayland_without_changing_qt_environment(self):
        env = KDE_WAYLAND.copy()
        self.assertEqual(session_diagnostics(env)["desktop_environment"], "kde-plasma")
        self.assertEqual(preview_diagnostics(True, env),
                         {"backend": "xwayland", "display_configured": True, "status": "ready_to_try"})
        self.assertEqual(env, KDE_WAYLAND)

    def test_colon_list_and_session_fallbacks_recognize_plasma(self):
        for env in ({"XDG_CURRENT_DESKTOP": "vendor:KDE"}, {"XDG_CURRENT_DESKTOP": " kde : plasma "},
                    {"XDG_SESSION_DESKTOP": "plasma"}, {"DESKTOP_SESSION": "plasmawayland"},
                    {"DESKTOP_SESSION": "plasmax11"}, {"KDE_FULL_SESSION": "true"},
                    {"KDE_FULL_SESSION": "1"}, {"XDG_CURRENT_DESKTOP": " ", "DESKTOP_SESSION": "plasma"}):
            with self.subTest(env=env):
                self.assertEqual(session_diagnostics(env)["desktop_environment"], "kde-plasma")

    def test_current_desktop_takes_precedence_over_stale_session_hints(self):
        session = session_diagnostics({"XDG_CURRENT_DESKTOP": "GNOME", "DESKTOP_SESSION": "plasma",
                                       "KDE_FULL_SESSION": "true"})
        self.assertEqual(session["desktop_environment"], "gnome")

    def test_gnome_and_unknown_desktops_need_no_special_preview_backend(self):
        for name, expected in (("GNOME", "gnome"), ("GNOME-Classic:GNOME", "gnome"),
                               ("XFCE", "other"), ("KDEish", "other"), ("", "unknown")):
            with self.subTest(name=name):
                env = {"XDG_CURRENT_DESKTOP": name, "DISPLAY": ":1"}
                self.assertEqual(session_diagnostics(env)["desktop_environment"], expected)
                self.assertEqual(preview_diagnostics(True, env)["backend"], "x11")

    def test_session_type_inference_and_explicit_tty(self):
        for env, expected in (({"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0"}, "wayland"),
                              ({"DISPLAY": ":0"}, "x11"), ({"XDG_SESSION_TYPE": " Wayland "}, "wayland"),
                              ({"XDG_SESSION_TYPE": "tty"}, "tty"), ({"DISPLAY": " "}, "unknown"),
                              ({}, "unknown")):
            with self.subTest(env=env):
                self.assertEqual(session_diagnostics(env)["session_type"], expected)

    def test_wayland_socket_alone_does_not_claim_tk_preview_is_ready(self):
        env = {**KDE_WAYLAND, "DISPLAY": ""}
        diagnostic = preview_diagnostics(True, env)
        self.assertEqual(diagnostic["status"], "xwayland_display_missing")
        self.assertFalse(diagnostic["display_configured"])
        self.assertIn("XWayland", preview_failure_hint(RuntimeError(), env))
        self.assertIn("Konsole", preview_failure_hint(RuntimeError(), env))

    def test_missing_tk_and_headless_display_have_distinct_guidance(self):
        self.assertEqual(preview_diagnostics(False, KDE_WAYLAND)["status"], "tkinter_missing")
        self.assertIn("python3-tkinter", preview_failure_hint(ModuleNotFoundError(), KDE_WAYLAND))
        self.assertEqual(preview_diagnostics(True, {})["status"], "display_missing")
        self.assertIn("desktop DISPLAY", preview_failure_hint(RuntimeError(), {}))

    def test_configured_but_inaccessible_display_has_connection_guidance(self):
        self.assertIn("could not initialize", preview_failure_hint(RuntimeError(), KDE_WAYLAND))

    def test_doctor_reports_plasma_prerequisites_without_preview_or_worker(self):
        output = io.StringIO()
        with patch.dict(os.environ, KDE_WAYLAND, clear=True), redirect_stdout(output), \
                patch("facepass.cli.Preview") as preview, \
                patch("facepass.supervisor.subprocess.Popen") as worker, \
                patch("facepass.cli.importlib.util.find_spec", return_value=object()):
            status = main(["--doctor", "--model", "/tmp/facepass-missing-model.task"])
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["desktop_environment"], "kde-plasma")
        self.assertEqual(result["session_type"], "wayland")
        self.assertEqual(result["preview"]["status"], "ready_to_try")
        self.assertFalse(result["authentication_enabled"])
        preview.assert_not_called()
        worker.assert_not_called()

    def test_plasma_and_headless_terminal_capture_do_not_require_tk_or_display(self):
        for env in (KDE_WAYLAND, {"XDG_CURRENT_DESKTOP": "KDE"}, {}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), \
                    redirect_stdout(io.StringIO()), patch("facepass.cli.Preview") as preview, \
                    patch("facepass.supervisor.supervise", return_value=failure_result("camera_open_failed")) as capture:
                status = main(["--duration", "0.5", "--json"])
                self.assertEqual(status, 2)
                capture.assert_called_once()
                preview.assert_not_called()
                self.assertTrue(capture.call_args.kwargs["pump"]())

    def test_preview_initialization_failure_never_starts_capture(self):
        for error, env, guidance in ((ImportError(), KDE_WAYLAND, "python3-tkinter"),
                                     (RuntimeError(), {**KDE_WAYLAND, "DISPLAY": ""}, "XWayland"),
                                     (RuntimeError(), KDE_WAYLAND, "could not initialize")):
            with self.subTest(guidance=guidance), patch.dict(os.environ, env, clear=True), \
                    redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors, \
                    patch("facepass.cli.Preview", side_effect=error), \
                    patch("facepass.supervisor.supervise") as capture:
                self.assertEqual(main(["--preview", "--json"]), 2)
                result = json.loads(output.getvalue())
                self.assertEqual(result["reason"], "preview_unavailable")
                self.assertEqual(result["confidence"], 0)
                self.assertIn(guidance, errors.getvalue())
                capture.assert_not_called()

    def test_terminal_final_explains_timeout_quality_and_shows_measurements(self):
        result = failure_result("observation_timeout")
        result["decision"] = "INSUFFICIENT_QUALITY"
        result["quality"].update(reasons=["blur"], blur_variance=12.5, mean_luminance=90.0, face_size_px=200.0)
        with redirect_stdout(io.StringIO()) as output, \
                patch("facepass.supervisor.supervise", return_value=result):
            self.assertEqual(main([]), 2)
        text = output.getvalue()
        self.assertIn("quality: blur", text)
        self.assertIn("blur_variance=12.50", text)
        self.assertIn("mean_luminance=90.00", text)
        self.assertIn("detector history was cleared", text)
        self.assertIn("clean camera lens", text)
        self.assertNotIn("liveness_detector_not_implemented", text)

    def test_terminal_reports_tolerated_quality_warnings(self):
        result = failure_result("observation_timeout")
        result["quality"].update(acceptable=True, reasons=[], warnings=["moderate_blur", "low_luminance"],
                                landmark_enhanced=True, contrast_span=10.0, evaluated_blur_variance=16.0)
        with redirect_stdout(io.StringIO()) as output, \
                patch("facepass.supervisor.supervise", return_value=result):
            self.assertEqual(main([]), 2)
        self.assertIn("Quality profile: tolerant", output.getvalue())
        self.assertIn("warnings: moderate_blur,low_luminance", output.getvalue())
        self.assertIn("landmark low-light enhancement: True", output.getvalue())

    def test_strict_profile_is_forwarded_to_the_capture_worker(self):
        with redirect_stdout(io.StringIO()), \
                patch("facepass.supervisor.supervise", return_value=failure_result("camera_open_failed")) as capture:
            self.assertEqual(main(["--quality-profile", "strict", "--json"]), 2)
        command = capture.call_args.args[0]
        self.assertEqual(command[command.index("--quality-profile") + 1], "strict")


if __name__ == "__main__":
    unittest.main()

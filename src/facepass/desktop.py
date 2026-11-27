"""Desktop diagnostics only; camera capture never requires a desktop session."""

from collections.abc import Mapping
import os


def session_diagnostics(environ: Mapping[str, str] | None = None) -> dict:
    """Read session hints without contacting a compositor or display server."""
    env = os.environ if environ is None else environ
    desktop = next((env.get(key, "").strip() for key in
                    ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION")
                    if env.get(key, "").strip()), "")
    names = {name.strip().lower() for name in desktop.split(":")}
    if names & {"kde", "plasma", "plasmawayland", "plasmax11"}:
        desktop_name = "kde-plasma"
    elif names & {"gnome", "gnome-classic", "gnome-xorg"}:
        desktop_name = "gnome"
    elif desktop:
        desktop_name = "other"
    elif env.get("KDE_FULL_SESSION", "").strip().lower() in {"true", "1"}:
        desktop_name = "kde-plasma"
    else:
        desktop_name = "unknown"

    wayland_display = bool(env.get("WAYLAND_DISPLAY", "").strip())
    x11_display = bool(env.get("DISPLAY", "").strip())
    session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
    if session_type not in {"wayland", "x11", "tty"}:
        session_type = "wayland" if wayland_display else "x11" if x11_display else "unknown"
    return {"desktop_environment": desktop_name, "session_type": session_type,
            "wayland_display_configured": wayland_display,
            "x11_display_configured": x11_display}


def preview_diagnostics(tk_available: bool, environ: Mapping[str, str] | None = None) -> dict:
    """Describe Tk's Linux X11/XWayland route, without opening a window.

    A configured DISPLAY is only a hint; connection/authentication can still fail.
    KDE/Qt environment variables are neither required nor modified.
    """
    session = session_diagnostics(environ)
    wayland = session["session_type"] == "wayland" or session["wayland_display_configured"]
    display = session["x11_display_configured"]
    backend = ("xwayland" if wayland else "x11") if display else "unavailable"
    if not tk_available:
        status = "tkinter_missing"
    elif not display:
        status = "xwayland_display_missing" if wayland else "display_missing"
    else:
        status = "ready_to_try"
    return {"backend": backend, "display_configured": display, "status": status}


def preview_failure_hint(error: Exception, environ: Mapping[str, str] | None = None) -> str:
    """Give actionable Fedora guidance while leaving terminal mode available."""
    session = session_diagnostics(environ)
    if isinstance(error, ImportError):
        hint = ("Preview needs Tk for this Python interpreter. With Fedora's system Python, "
                "install it with: sudo dnf install python3-tkinter.")
    elif not session["x11_display_configured"]:
        if session["session_type"] == "wayland" or session["wayland_display_configured"]:
            hint = ("Tk preview needs XWayland and a DISPLAY in this Wayland session. "
                    "Check Fedora's xorg-x11-server-Xwayland package and run from Konsole "
                    "in KDE Plasma or GNOME Terminal in GNOME.")
        else:
            hint = ("Tk preview needs a desktop DISPLAY. Run from Konsole in KDE Plasma "
                    "or GNOME Terminal in GNOME, inside your logged-in desktop session.")
    else:
        hint = ("Tk could not initialize the desktop display. Run from Konsole in KDE Plasma "
                "or GNOME Terminal in GNOME; check Tk and access to your session's "
                "X11/XWayland display.")
    return hint + " Run without --preview for terminal capture; --doctor reports preview prerequisites."

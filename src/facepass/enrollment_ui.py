"""Ordinary-user enrollment UI; administrator authorization stays in PolicyKit."""

import argparse
import os
import select
import signal
import subprocess
import time

from .auth_cli import SYSTEM_PYTHON, status
from .evaluation import strict_json


class EnrollmentWindow:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("FacePass — Set up face unlock")
        self.root.minsize(660, 620)
        self.process = None
        self.buffer = bytearray()
        self.final = None
        self.image = None
        self.started = None
        ttk.Label(self.root, text="Set up FacePass", font=("sans", 20)).pack(pady=(18, 8))
        ttk.Label(self.root, text="Enroll your face, then follow a fresh verification challenge.\n"
                  "Only private face templates are saved locally. Camera images are not saved.",
                  justify="center").pack(padx=20, pady=8)
        ttk.Label(self.root, text="Experimental RGB face unlock can be fooled by sophisticated replays.\n"
                  "Your password remains available.", justify="center").pack(pady=6)
        self.preview = ttk.Label(self.root, text="Camera preview appears after administrator authorization.", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.message = ttk.Label(self.root, text="Ready to enroll", wraplength=630, anchor="center", justify="center")
        self.message.pack(fill="x", padx=16, pady=8)
        self.replace = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.root, text="Replace an existing enrollment", variable=self.replace).pack()
        buttons = ttk.Frame(self.root)
        buttons.pack(pady=14)
        self.enroll_button = ttk.Button(buttons, text="Enroll and verify", command=self.start)
        self.enroll_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="left", padx=6)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        current = status(os.getuid())
        if not current["installed"]:
            self.enroll_button.configure(state="disabled")
            self.message.configure(text="Install the system authentication component first using the prepared installer.")
        elif current["enrolled"]:
            self.message.configure(text="This account is already enrolled. Choose Replace enrollment to update it.")
        self.root.after(50, self.pump)

    def start(self):
        if self.process is not None:
            return
        command = ["/usr/bin/pkexec", str(SYSTEM_PYTHON), "-I", "-m", "facepass.auth_cli",
                   "enroll", "--uid", str(os.getuid()), "--preview"]
        if self.replace.get():
            command.append("--replace")
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, start_new_session=True)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.started = time.monotonic()
        self.final = None
        self.buffer.clear()
        self.enroll_button.configure(state="disabled")
        self.message.configure(text="Complete the administrator dialog, then look at the camera and follow the instructions.")

    def _message(self, value):
        if value.get("kind") == "preview" and isinstance(value.get("png"), str):
            self.image = self.tk.PhotoImage(data=value["png"])
            self.preview.configure(image=self.image, text="")
        elif value.get("kind") == "info":
            self.message.configure(text=value["message"])
        elif value.get("kind") == "final":
            self.final = value

    def pump(self):
        if self.process is not None:
            try:
                fd = self.process.stdout.fileno()
                for _ in range(8):
                    if not select.select([fd], [], [], 0)[0]:
                        break
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    self.buffer.extend(chunk)
                    if len(self.buffer) > 2_000_000:
                        raise ValueError("oversized enrollment response")
                    while b"\n" in self.buffer:
                        line, _, remainder = self.buffer.partition(b"\n")
                        self.buffer = bytearray(remainder)
                        self._message(strict_json(line))
                if time.monotonic() - self.started > 180:
                    self.cancel()
                    self.message.configure(text="Enrollment timed out. You can retry; password login remains available.")
                elif self.process.poll() is not None:
                    success = self.process.returncode == 0 and self.final and self.final.get("success") is True
                    self.process.stdout.close()
                    self.process = None
                    self.enroll_button.configure(state="normal")
                    self.preview.configure(image="", text="Camera closed.")
                    self.image = None
                    if success:
                        self.message.configure(text="Face unlock is ready. Lock your screen and choose FacePass.\n"
                                               "Follow the blink and head-turn instructions. You can always use your password.")
                    else:
                        self.message.configure(text=(self.final or {}).get("message", "Enrollment was not saved. Retry with your face closer to the camera and follow each instruction."))
            except Exception:
                self.cancel()
                self.message.configure(text="Enrollment could not finish. Password login remains available.")
        self.root.after(50, self.pump)

    def cancel(self):
        if self.process is not None:
            # pkexec's root child has a separate supervisor deadline as well.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            self.process.stdout.close()
            self.process = None
            self.enroll_button.configure(state="normal")
        self.preview.configure(image="", text="Camera preview stopped.")
        self.image = None

    def close(self):
        self.cancel()
        self.root.destroy()


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if os.geteuid() == 0:
        raise SystemExit("Run FacePass Setup as your desktop user.")
    window = EnrollmentWindow()
    window.root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

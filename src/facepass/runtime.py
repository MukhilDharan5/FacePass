"""Validated capture configuration shared by the CLI and local research service."""

from dataclasses import dataclass
import math
import re
import sys

MAX_CAPTURE_FRAMES = 1100  # 35 seconds at 30 fps, with room for a boundary frame.
QUALITY_PROFILES = ("tolerant", "strict")


@dataclass(frozen=True)
class CaptureOptions:
    device: str = "/dev/video0"
    duration: float = 5.0
    fps: int = 12
    fixed_window: bool = False
    extension_duration: float = 2.0
    quality_profile: str = "tolerant"

    def __post_init__(self):
        if not isinstance(self.device, str) or re.fullmatch(r"/dev/video[0-9]{1,6}", self.device) is None:
            raise ValueError("device must be a local /dev/videoN capture node")
        for name, low, high in (("duration", .5, 30), ("extension_duration", 0, 5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be {low}..{high} seconds")
        if type(self.fps) is not int or not 5 <= self.fps <= 30:
            raise ValueError("fps must be an integer in 5..30")
        if type(self.fixed_window) is not bool:
            raise ValueError("fixed_window must be boolean")
        if not isinstance(self.quality_profile, str) or self.quality_profile not in QUALITY_PROFILES:
            raise ValueError("quality_profile must be tolerant or strict")

    @property
    def maximum_extension(self):
        return 0.0 if self.fixed_window else self.extension_duration

    def worker_command(self, model):
        command = [sys.executable, "-m", "facepass", "--worker", "--model", str(model),
                   "--device", self.device, "--duration", str(self.duration), "--fps", str(self.fps),
                   "--extension-duration", str(self.extension_duration),
                   "--quality-profile", self.quality_profile]
        if self.fixed_window:
            command.append("--fixed-window")
        return command


def add_capture_arguments(parser):
    parser.add_argument("--device", default="/dev/video0", help="Local /dev/videoN camera")
    parser.add_argument("--duration", type=float, default=5.0, help="Base observation limit, 0.5..30 seconds")
    parser.add_argument("--fps", type=int, default=12, help="Processing rate, 5..30 frames/second")
    parser.add_argument("--fixed-window", action="store_true", help="Disable early completion and evidence extensions")
    parser.add_argument("--extension-duration", type=float, default=2.0, help="Maximum one-time extension, 0..5 seconds")
    parser.add_argument("--quality-profile", choices=QUALITY_PROFILES, default="tolerant",
                        help="Capture usability: tolerant (default) accepts moderate blur/dim detail; strict uses the original gates")


def options_from_args(args):
    return CaptureOptions(**{name: getattr(args, name) for name in CaptureOptions.__dataclass_fields__})

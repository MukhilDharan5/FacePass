"""Bounded scalar timing summaries. Measurement does not change detector policy."""

import resource
import time

from .runtime import MAX_CAPTURE_FRAMES

def distribution(values):
    samples = sorted(values)
    if not samples:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "maximum": None}

    def percentile(fraction):
        position = (len(samples) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        return round(samples[lower] + (samples[upper] - samples[lower]) * (position - lower), 3)

    return {"count": len(samples), "p50": percentile(.50), "p95": percentile(.95),
            "p99": percentile(.99), "maximum": round(samples[-1], 3)}


class CaptureProfile:
    """End-to-end harness timing, including setup and supervised worker CPU.

    Child CPU is valid for an isolated harness process. RSS is the OS child peak,
    not a difference between peaks and not a per-frame allocation measurement.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.cpu_started = self._cpu()
        self.first_update = self.last_update = None
        self.intervals = []
        self.frames = 0

    @staticmethod
    def _cpu():
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime

    def update(self, result):
        now = self.clock()
        if self.first_update is None:
            self.first_update = now
        if self.last_update is not None and len(self.intervals) < MAX_CAPTURE_FRAMES:
            self.intervals.append(max(0, (now - self.last_update) * 1000))
        self.last_update = now
        self.frames = max(self.frames, result["frame_count"])

    def finish(self, result):
        elapsed = max(0, self.clock() - self.started)
        cpu = max(0, self._cpu() - self.cpu_started)
        frames = max(self.frames, result["frame_count"])
        return {"schema_version": 1, "workload": "supervised_camera_capture", "authentication_enabled": False,
                "wall_ms": round(elapsed * 1000, 3), "cpu_ms": round(cpu * 1000, 3),
                "average_cpu_cores": round(cpu / elapsed, 3) if elapsed else None,
                "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                "first_update_ms": round((self.first_update - self.started) * 1000, 3) if self.first_update is not None else None,
                "frames": frames, "processed_fps_including_startup": round(frames / elapsed, 3) if elapsed else None,
                "interupdate_ms": distribution(self.intervals), "final_reason": result["reason"]}

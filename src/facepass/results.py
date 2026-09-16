"""Public diagnostics contain scalar measurements, never landmark vectors."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math


class Decision(str, Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"


SIGNAL_NAMES = (
    "blink", "gaze", "head_motion", "facial_geometry",
    "illumination_consistency", "replay_artifact_detection", "temporal_consistency",
)


@dataclass(frozen=True)
class SignalResult:
    confidence: float = 0.5
    reliability: float = 0.0
    available: bool = False
    diagnostic_information: str = "liveness_detector_not_implemented"

    def __post_init__(self):
        for value in (self.confidence, self.reliability):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid signal score")
        if not self.available and self.reliability != 0:
            raise ValueError("unavailable evidence cannot have reliability")


@dataclass(frozen=True)
class QualityResult:
    acceptable: bool = False
    face_size_px: float | None = None
    face_area_fraction: float | None = None
    blur_variance: float | None = None
    mean_luminance: float | None = None
    dark_fraction: float | None = None
    bright_fraction: float | None = None
    landmark_confidence: float | None = None
    landmark_valid_fraction: float | None = None
    orientation_acceptable: bool | None = None
    occlusion: str = "not_measured"
    reasons: tuple[str, ...] = ("not_observed",)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite quality measurement: {name}")


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    STABLE = "STABLE"
    LOST = "LOST"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class TrackingResult:
    state: TrackingState = TrackingState.SEARCHING
    track_id: int | None = None  # Session-local sequence number, never an identity.
    continuous_frames: int = 0
    history_frame_count: int = 0
    track_duration_ms: int = 0
    continuity_score: float = 0.0  # Heuristic margin, not an identity probability.
    appearance_available: bool = False
    center_displacement: float | None = None
    scale_change: float | None = None
    shape_residual: float | None = None
    appearance_change: float | None = None
    reset_count: int = 0
    history_reset: bool = False
    event: str = "not_observed"
    reset_reason: str = "none"

    def __post_init__(self):
        if not isinstance(self.state, TrackingState):
            raise ValueError("invalid tracking state")
        for value in (self.continuity_score, self.center_displacement, self.scale_change,
                      self.shape_residual, self.appearance_change):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid tracking measurement")
        if self.continuity_score > 1:
            raise ValueError("invalid tracking score")


@dataclass(frozen=True)
class MotionDiagnostics:
    status: str = "not_observed"
    historical: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    pose_available: bool = False
    yaw_degrees: float | None = None
    pitch_degrees: float | None = None
    roll_degrees: float | None = None
    pose_reprojection_error_px: float | None = None
    rotation_span_degrees: float | None = None
    temporal_coherence: float = 0.0
    landmark_motion_px: float | None = None
    background_available: bool = False
    background_point_count: int = 0
    background_inlier_fraction: float = 0.0
    common_motion_px: float | None = None
    independent_motion_px: float | None = None
    common_motion_suppressed: bool = False
    planar_available: bool = False
    planar_holdout_error_px: float | None = None
    nonplanar_fraction: float | None = None

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite motion measurement")


@dataclass(frozen=True)
class LivenessResult:
    decision: Decision = Decision.UNCERTAIN
    confidence: float = 0.0
    signals: dict[str, SignalResult] = field(
        default_factory=lambda: {name: SignalResult() for name in SIGNAL_NAMES}
    )
    quality: QualityResult = field(default_factory=QualityResult)
    tracking: TrackingResult = field(default_factory=TrackingResult)
    motion: MotionDiagnostics = field(default_factory=MotionDiagnostics)
    flags: tuple[str, ...] = ("liveness_not_implemented",)
    reason: str = "not_started"
    observation_duration_ms: int = 0
    frame_count: int = 0
    face_count: int = 0
    landmark_count: int = 0
    terminal: bool = False

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid liveness confidence")
        if self.frame_count < 0 or self.observation_duration_ms < 0:
            raise ValueError("invalid observation counters")

    def to_dict(self):
        value = asdict(self)
        value["decision"] = self.decision.value
        value["tracking"]["state"] = self.tracking.state.value
        return value


def failure_result(reason: str) -> dict:
    return LivenessResult(reason=reason, terminal=True).to_dict()

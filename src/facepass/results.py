"""Public diagnostics contain scalar measurements, never landmark vectors."""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import math


def _measurements(instance):
    # These diagnostic records are shallow scalar data. Validation does not need
    # asdict's recursive allocation/copy; public serialization still uses asdict.
    return ((item.name, getattr(instance, item.name)) for item in fields(instance))


class Decision(str, Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"


SIGNAL_NAMES = (
    "blink", "gaze", "head_motion", "facial_geometry",
    "illumination_consistency", "replay_artifact_detection", "temporal_consistency",
)
BLINK_STATES = ("NOT_OBSERVED", "ACQUIRING_OPEN", "OPEN", "CLOSING", "CLOSED",
                "OPENING", "SUSTAINED_CLOSED")


@dataclass(frozen=True)
class SignalResult:
    confidence: float = 0.5
    reliability: float = 0.0
    available: bool = False
    diagnostic_information: str = "not_observed"

    def __post_init__(self):
        for value in (self.confidence, self.reliability):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid signal score")
        if not self.available and self.reliability != 0:
            raise ValueError("unavailable evidence cannot have reliability")


@dataclass(frozen=True)
class QualityResult:
    acceptable: bool = False
    profile: str = "tolerant"
    warnings: tuple[str, ...] = ()
    landmark_enhanced: bool = False
    face_size_px: float | None = None
    face_area_fraction: float | None = None
    blur_variance: float | None = None
    evaluated_blur_variance: float | None = None
    contrast_span: float | None = None
    mean_luminance: float | None = None
    dark_fraction: float | None = None
    bright_fraction: float | None = None
    landmark_confidence: float | None = None
    landmark_valid_fraction: float | None = None
    orientation_acceptable: bool | None = None
    occlusion: str = "not_measured"
    reasons: tuple[str, ...] = ("not_observed",)

    def __post_init__(self):
        for name, value in _measurements(self):
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
        for _, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite motion measurement")


@dataclass(frozen=True)
class BlinkDiagnostics:
    status: str = "not_observed"
    historical: bool = False
    state: str = "NOT_OBSERVED"
    eyes_available: bool = False
    left_ear: float | None = None
    right_ear: float | None = None
    left_eye_width_px: float | None = None
    right_eye_width_px: float | None = None
    observed_frame_count: int = 0
    observation_duration_ms: int = 0
    blink_count: int = 0
    rejected_sequence_count: int = 0
    closed_duration_ms: int | None = None
    last_blink_duration_ms: int | None = None
    last_blink_age_ms: int | None = None
    event: str = "not_observed"

    def __post_init__(self):
        if self.state not in BLINK_STATES:
            raise ValueError("invalid blink state")
        for _, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite blink measurement")


@dataclass(frozen=True)
class GazeDiagnostics:
    status: str = "not_observed"
    historical: bool = False
    left_eye_available: bool = False
    right_eye_available: bool = False
    pose_available: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    left_horizontal_ratio: float | None = None
    right_horizontal_ratio: float | None = None
    left_vertical_ratio: float | None = None
    right_vertical_ratio: float | None = None
    left_eye_width_px: float | None = None
    right_eye_width_px: float | None = None
    left_iris_diameter_px: float | None = None
    right_iris_diameter_px: float | None = None
    horizontal_span: float | None = None
    vertical_span: float | None = None
    robust_displacement: float | None = None
    motion_coherence: float = 0.0
    bilateral_agreement: float = 0.0
    head_rotation_span_degrees: float | None = None
    head_motion_suppressed: bool = False
    independent_motion_observed: bool = False
    event: str = "not_observed"

    def __post_init__(self):
        for _, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite gaze measurement")


@dataclass(frozen=True)
class IlluminationDiagnostics:
    status: str = "not_observed"
    historical: bool = False
    face_regions_available: bool = False
    ambient_available: bool = False
    pose_available: bool = False
    motion_context_available: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    face_region_count: int = 0
    ambient_region_count: int = 0
    minimum_region_pixels: int = 0
    clipped_fraction: float = 0.0
    ambient_luminance: float | None = None
    ambient_red_balance: float | None = None
    ambient_green_balance: float | None = None
    ambient_blue_balance: float | None = None
    cheek_log_contrast: float | None = None
    forehead_chin_log_contrast: float | None = None
    nose_cheek_log_contrast: float | None = None
    socket_log_contrast: float | None = None
    global_gain_ev: float | None = None
    common_color_shift: float | None = None
    ambient_spatial_change: float | None = None
    face_spatial_change: float | None = None
    regional_response_residual: float | None = None
    changing_face_regions: int = 0
    response_coherence: float = 0.0
    direction_agreement: float = 0.0
    color_response_error: float | None = None
    ambient_response_direction_degrees: float | None = None
    face_response_direction_degrees: float | None = None
    head_rotation_span_degrees: float | None = None
    face_motion_px: float | None = None
    background_motion_px: float | None = None
    motion_suppressed: bool = False
    response_mismatch: bool = False
    consistent_response_observed: bool = False
    event: str = "not_observed"

    def __post_init__(self):
        for _, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite illumination measurement")


@dataclass(frozen=True)
class ReplayDiagnostics:
    status: str = "not_observed"
    historical: bool = False
    regions_available: bool = False
    background_available: bool = False
    motion_context_available: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    face_region_count: int = 0
    background_region_count: int = 0
    minimum_patch_width_px: int = 0
    texture_contrast: float | None = None
    periodic_peak_score: float | None = None
    chroma_grid_score: float | None = None
    banding_score: float | None = None
    sharpening_score: float | None = None
    resampling_score: float | None = None
    glare_score: float | None = None
    flat_edge_score: float | None = None
    background_artifact_score: float | None = None
    affected_face_regions: int = 0
    artifact_family_count: int = 0
    artifact_persistence: float = 0.0
    temporal_score_variation: float | None = None
    face_motion_px: float | None = None
    background_motion_px: float | None = None
    motion_suppressed: bool = False
    replay_artifact_observed: bool = False
    event: str = "not_observed"

    def __post_init__(self):
        for _, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite replay measurement")


@dataclass(frozen=True)
class TemporalDiagnostics:
    """Bounded cross-signal observations; this is not a calibrated score."""

    status: str = "not_observed"
    historical: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    landmark_group_available: bool = False
    photometric_group_available: bool = False
    replay_group_available: bool = False
    motion_context_available: bool = False
    landmark_group_support: float = 0.0
    photometric_group_support: float = 0.0
    replay_suspicion: float = 0.0
    positive_group_count: int = 0
    supporting_signal_count: int = 0
    support_overlap: float = 0.0
    cross_group_agreement: float = 0.0
    temporal_stability: float = 0.0
    conflict_count: int = 0
    conflict_observed: bool = False
    dependence_adjusted_support: float = 0.0
    missing_group_count: int = 3
    motion_suppressed: bool = False
    consistent_behavior_observed: bool = False
    event: str = "not_observed"

    def __post_init__(self):
        for name, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite temporal measurement: {name}")


@dataclass(frozen=True)
class FusionDiagnostics:
    """Dependence-aware group summary retained for review and evaluation only."""

    status: str = "not_observed"
    historical: bool = False
    window_frame_count: int = 0
    window_duration_ms: int = 0
    landmark_group_score: float = 0.0
    photometric_group_score: float = 0.0
    replay_group_score: float = 0.0
    active_group_count: int = 0
    available_group_count: int = 0
    missing_group_count: int = 3
    positive_group_count: int = 0
    suspicious_group_count: int = 0
    conflict_count: int = 0
    dependence_penalty: float = 0.0
    raw_combined_score: float = 0.0
    adjusted_combined_score: float = 0.0
    coverage: float = 0.0
    group_agreement: float = 0.0
    individual_signal_count: int = 0
    liveness_candidate: bool = False
    spoof_candidate: bool = False
    event: str = "not_observed"

    def __post_init__(self):
        for name, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite fusion measurement: {name}")


@dataclass(frozen=True)
class ObservationDiagnostics:
    """Bounded adaptive-observation policy state, never an auth decision."""

    status: str = "not_observed"
    historical: bool = False
    elapsed_ms: int = 0
    minimum_duration_ms: int = 1500
    base_limit_ms: int = 5000
    effective_limit_ms: int = 5000
    extension_limit_ms: int = 2000
    extension_count: int = 0
    quality_frame_count: int = 0
    stable_frame_count: int = 0
    quality_frame_fraction: float = 0.0
    temporal_ready: bool = False
    positive_group_count: int = 0
    conflict_count: int = 0
    early_completion: bool = False
    extension_requested: bool = False
    challenge_state: str = "disabled"
    event: str = "not_observed"

    def __post_init__(self):
        for name, value in _measurements(self):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite observation measurement: {name}")


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
    blink: BlinkDiagnostics = field(default_factory=BlinkDiagnostics)
    gaze: GazeDiagnostics = field(default_factory=GazeDiagnostics)
    illumination: IlluminationDiagnostics = field(default_factory=IlluminationDiagnostics)
    replay: ReplayDiagnostics = field(default_factory=ReplayDiagnostics)
    temporal: TemporalDiagnostics = field(default_factory=TemporalDiagnostics)
    fusion: FusionDiagnostics = field(default_factory=FusionDiagnostics)
    observation: ObservationDiagnostics = field(default_factory=ObservationDiagnostics)
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

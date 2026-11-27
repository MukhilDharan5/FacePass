"""Bound model/camera hangs and validate diagnostics before exposing them."""

import json
import math
import os
import re
import selectors
import subprocess
import time

from .results import BLINK_STATES, SIGNAL_NAMES, TrackingState, failure_result
from .runtime import MAX_CAPTURE_FRAMES


MAX_MESSAGE_BYTES = 2_000_000


def _code(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,80}", value) is not None


def _number(value, low=0, high=1):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and low <= value <= high)


def validate_result(value):
    template = failure_result("template")
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError("unknown result schema")
    # Fusion remains diagnostic only: individual or combined evidence can never
    # enable LIVE/SPOOF in this milestone.
    if (value["decision"] not in ("UNCERTAIN", "INSUFFICIENT_QUALITY")
            or not _number(value["confidence"]) or value["confidence"] != 0.0):
        raise ValueError("unsupported decision")
    if not _code(value["reason"]) or not isinstance(value["terminal"], bool):
        raise ValueError("invalid result state")
    for key, maximum in (("frame_count", MAX_CAPTURE_FRAMES), ("face_count", 4), ("landmark_count", 478),
                         ("observation_duration_ms", 60_000)):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or not 0 <= value[key] <= maximum:
            raise ValueError("invalid result counter")
    if (not isinstance(value["flags"], list) or len(value["flags"]) > 20
            or not all(_code(flag) for flag in value["flags"])
            or "liveness_not_implemented" not in value["flags"]):
        raise ValueError("invalid flags")
    scores = value["signals"]
    if not isinstance(scores, dict) or set(scores) != set(SIGNAL_NAMES):
        raise ValueError("unknown signals")
    for name, score in scores.items():
        if (not isinstance(score, dict) or set(score) != set(template["signals"]["blink"])
                or not _number(score["confidence"]) or not _number(score["reliability"])
                or not isinstance(score["available"], bool)
                or not _code(score["diagnostic_information"])):
            raise ValueError("invalid evidence")
        if not score["available"] and score["reliability"] != 0:
            raise ValueError("unavailable evidence cannot have reliability")
        if name not in ("head_motion", "facial_geometry", "blink", "gaze", "illumination_consistency",
                        "replay_artifact_detection", "temporal_consistency") and (score["available"] or score["reliability"] != 0):
            raise ValueError("unimplemented detector evidence")
        if name in ("head_motion", "facial_geometry") and (score["confidence"] > 0.68 + 1e-9 or score["reliability"] > 0.4):
            raise ValueError("motion evidence exceeds experimental caps")
        if name == "blink" and (score["confidence"] > 0.65 + 1e-9 or score["reliability"] > 0.35):
            raise ValueError("blink evidence exceeds experimental caps")
        if name == "gaze" and (score["confidence"] > 0.62 + 1e-9 or score["reliability"] > 0.30):
            raise ValueError("gaze evidence exceeds experimental caps")
        if name == "illumination_consistency" and (score["confidence"] > 0.58 + 1e-9 or score["reliability"] > 0.20):
            raise ValueError("illumination evidence exceeds experimental caps")
        if name == "replay_artifact_detection" and (score["confidence"] > 0.5 or score["reliability"] > 0.25):
            raise ValueError("replay evidence exceeds experimental caps")
        if name == "temporal_consistency" and (score["confidence"] > 0.56 + 1e-9 or score["reliability"] > 0.18):
            raise ValueError("temporal evidence exceeds experimental caps")
    quality = value["quality"]
    if not isinstance(quality, dict) or set(quality) != set(template["quality"]):
        raise ValueError("unknown quality fields")
    for key, measurement in quality.items():
        if key in ("acceptable", "landmark_enhanced"):
            if not isinstance(measurement, bool):
                raise ValueError("invalid quality state")
        elif key == "orientation_acceptable":
            if measurement is not None and not isinstance(measurement, bool):
                raise ValueError("invalid orientation")
        elif key == "occlusion":
            if measurement != "not_measured":
                raise ValueError("unsupported occlusion evidence")
        elif key == "profile":
            if measurement not in ("tolerant", "strict"):
                raise ValueError("invalid quality profile")
        elif key in ("reasons", "warnings"):
            if not isinstance(measurement, list) or len(measurement) > 20 or not all(_code(r) for r in measurement):
                raise ValueError("invalid quality reasons")
        elif measurement is not None and not _number(measurement, 0,
                {"blur_variance": 1_100_000, "evaluated_blur_variance": 20_000_000}.get(key, 1_000_000)):
            raise ValueError("invalid quality measurement")
    tracking = value["tracking"]
    if not isinstance(tracking, dict) or set(tracking) != set(template["tracking"]):
        raise ValueError("unknown tracking fields")
    if tracking["state"] not in {state.value for state in TrackingState}:
        raise ValueError("unknown tracking state")
    active = tracking["state"] in ("ACQUIRING", "STABLE")
    track_id = tracking["track_id"]
    if active:
        if not isinstance(track_id, int) or isinstance(track_id, bool) or not 1 <= track_id <= MAX_CAPTURE_FRAMES:
            raise ValueError("invalid track identifier")
        if value["terminal"] or value["face_count"] != 1 or not quality["acceptable"]:
            raise ValueError("invalid active track")
    elif track_id is not None:
        raise ValueError("inactive track cannot identify a current track")
    for key, maximum in (("continuous_frames", MAX_CAPTURE_FRAMES), ("history_frame_count", 128),
                         ("track_duration_ms", 60_000), ("reset_count", MAX_CAPTURE_FRAMES)):
        if not isinstance(tracking[key], int) or isinstance(tracking[key], bool) or not 0 <= tracking[key] <= maximum:
            raise ValueError("invalid tracking counter")
    if active:
        if not 1 <= tracking["history_frame_count"] <= tracking["continuous_frames"]:
            raise ValueError("invalid tracking history")
        if tracking["state"] == "STABLE" and tracking["continuous_frames"] < 3:
            raise ValueError("invalid stable track")
    elif any(tracking[key] for key in ("history_frame_count", "continuous_frames", "track_duration_ms")):
        raise ValueError("inactive track retains history")
    for key in ("appearance_available", "history_reset"):
        if not isinstance(tracking[key], bool):
            raise ValueError("invalid tracking availability")
    if not _code(tracking["event"]) or not _code(tracking["reset_reason"]):
        raise ValueError("invalid tracking event")
    if not _number(tracking["continuity_score"]):
        raise ValueError("invalid tracking score")
    for key, maximum in (("center_displacement", 100), ("scale_change", 100),
                         ("shape_residual", 2), ("appearance_change", 1)):
        if tracking[key] is not None and not _number(tracking[key], 0, maximum):
            raise ValueError("invalid tracking measurement")
    motion = value["motion"]
    if not isinstance(motion, dict) or set(motion) != set(template["motion"]):
        raise ValueError("unknown motion fields")
    for key, measurement in motion.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid motion status")
        elif isinstance(template["motion"][key], bool):
            if not isinstance(measurement, bool):
                raise ValueError("invalid motion availability")
        elif key in ("window_frame_count", "window_duration_ms", "background_point_count"):
            limit = {"window_frame_count": 64, "window_duration_ms": 2000, "background_point_count": 80}[key]
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= limit:
                raise ValueError("invalid motion counter")
        elif measurement is not None:
            low, high = (-180, 180) if key.endswith("degrees") else (0, 100_000)
            if key in ("temporal_coherence", "background_inlier_fraction", "nonplanar_fraction"):
                low, high = 0, 1
            if not _number(measurement, low, high):
                raise ValueError("invalid motion measurement")
    if motion["historical"]:
        if not value["terminal"] or tracking["state"] != "CLOSED" or motion["status"] != "historical_diagnostics":
            raise ValueError("invalid historical motion diagnostics")
    for name in ("head_motion", "facial_geometry"):
        if scores[name]["available"] and (value["terminal"] or tracking["state"] != "STABLE"
                                         or not quality["acceptable"] or motion["historical"]
                                         or motion["window_frame_count"] < 6 or motion["window_duration_ms"] < 450):
            raise ValueError("motion evidence outside valid observation")
    blink = value["blink"]
    if not isinstance(blink, dict) or set(blink) != set(template["blink"]):
        raise ValueError("unknown blink fields")
    for key, measurement in blink.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid blink status")
        elif key == "state":
            if measurement not in BLINK_STATES:
                raise ValueError("invalid blink state")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid blink event")
        elif key in ("historical", "eyes_available"):
            if not isinstance(measurement, bool):
                raise ValueError("invalid blink availability")
        elif key.endswith("ear") or key.endswith("width_px"):
            if measurement is not None and not _number(measurement, 0, 0.6 if key.endswith("ear") else 3840):
                raise ValueError("invalid eye measurement")
        elif measurement is not None:
            limit = MAX_CAPTURE_FRAMES if key in ("observed_frame_count", "blink_count", "rejected_sequence_count") else 60_000
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= limit:
                raise ValueError("invalid blink counter")
        elif template["blink"][key] is not None:
            raise ValueError("missing blink counter")
    if blink["historical"] != (blink["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical blink diagnostics")
    if blink["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical blink diagnostics")
    if blink["eyes_available"] and any(blink[key] is None for key in
                                      ("left_ear", "right_ear", "left_eye_width_px", "right_eye_width_px")):
        raise ValueError("incomplete eye measurements")
    score = scores["blink"]
    if score["available"] and (value["terminal"] or tracking["state"] != "STABLE" or not quality["acceptable"]
                               or blink["historical"] or blink["status"] != "observing" or not blink["eyes_available"]
                               or min(blink["left_eye_width_px"], blink["right_eye_width_px"]) < 18
                               or blink["observed_frame_count"] < 3 or blink["observation_duration_ms"] < 150):
        raise ValueError("blink evidence outside valid observation")
    if score["available"] and score["confidence"] > 0.5:
        if (blink["state"] != "OPEN" or blink["blink_count"] < 1
                or min(blink["left_ear"], blink["right_ear"]) < 0.23
                or blink["last_blink_duration_ms"] is None or not 60 <= blink["last_blink_duration_ms"] <= 650
                or blink["last_blink_age_ms"] is None or blink["last_blink_age_ms"] > 2000):
            raise ValueError("positive blink evidence without a fresh completed sequence")
    gaze = value["gaze"]
    if not isinstance(gaze, dict) or set(gaze) != set(template["gaze"]):
        raise ValueError("unknown gaze fields")
    for key, measurement in gaze.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid gaze status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid gaze event")
        elif isinstance(template["gaze"][key], bool):
            if not isinstance(measurement, bool):
                raise ValueError("invalid gaze availability")
        elif key in ("window_frame_count", "window_duration_ms"):
            limit = 64 if key == "window_frame_count" else 2000
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= limit:
                raise ValueError("invalid gaze counter")
        else:
            low, high = 0, 1
            if key.endswith("vertical_ratio"):
                low, high = -0.25, 0.25
            elif key.endswith("_px"):
                high = 3840
            elif key.endswith("degrees"):
                high = 180
            if measurement is None:
                if template["gaze"][key] is not None:
                    raise ValueError("missing gaze measurement")
            elif not _number(measurement, low, high):
                raise ValueError("invalid gaze measurement")
    if gaze["historical"] != (gaze["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical gaze diagnostics")
    if gaze["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical gaze diagnostics")
    for side in ("left", "right"):
        if gaze[side + "_eye_available"]:
            if any(gaze[side + "_" + key] is None for key in
                   ("horizontal_ratio", "vertical_ratio", "eye_width_px", "iris_diameter_px")):
                raise ValueError("incomplete iris measurements")
            if gaze[side + "_eye_width_px"] < 28 or gaze[side + "_iris_diameter_px"] < 6:
                raise ValueError("insufficient iris resolution")
    score = scores["gaze"]
    if score["available"] and (value["terminal"] or tracking["state"] != "STABLE" or not quality["acceptable"]
                               or gaze["historical"] or gaze["status"] != "observing"
                               or not gaze["left_eye_available"] or not gaze["right_eye_available"]
                               or not gaze["pose_available"] or gaze["window_frame_count"] < 6
                               or gaze["window_duration_ms"] < 450):
        raise ValueError("gaze evidence outside valid observation")
    if gaze["independent_motion_observed"] or score["available"] and score["confidence"] > 0.5:
        if (not gaze["independent_motion_observed"] or gaze["head_motion_suppressed"]
                or not gaze["pose_available"] or not gaze["left_eye_available"] or not gaze["right_eye_available"]
                or gaze["window_frame_count"] < 6 or gaze["window_duration_ms"] < 450
                or gaze["head_rotation_span_degrees"] is None or gaze["head_rotation_span_degrees"] > 1.5
                or gaze["robust_displacement"] is None
                or gaze["robust_displacement"] < max(0.035, 2 / min(gaze["left_eye_width_px"], gaze["right_eye_width_px"]))
                or gaze["motion_coherence"] < 0.70 or gaze["bilateral_agreement"] < 0.70):
            raise ValueError("positive gaze evidence without independent bilateral motion")
    illumination = value["illumination"]
    if not isinstance(illumination, dict) or set(illumination) != set(template["illumination"]):
        raise ValueError("unknown illumination fields")
    counter_limits = {"window_frame_count": 64, "window_duration_ms": 2000, "face_region_count": 7,
                      "ambient_region_count": 4, "minimum_region_pixels": 76800, "changing_face_regions": 7}
    for key, measurement in illumination.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid illumination status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid illumination event")
        elif isinstance(template["illumination"][key], bool):
            if not isinstance(measurement, bool):
                raise ValueError("invalid illumination availability")
        elif key in counter_limits:
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= counter_limits[key]:
                raise ValueError("invalid illumination counter")
        else:
            low, high = 0, 1
            if key.endswith("log_contrast"):
                low, high = -6, 6
            elif key == "global_gain_ev":
                low, high = -8, 8
            elif key.endswith("direction_degrees"):
                low, high = -180, 180
            elif key == "head_rotation_span_degrees":
                high = 180
            elif key.endswith("motion_px"):
                high = 10000
            elif key in ("common_color_shift", "ambient_spatial_change", "face_spatial_change", "regional_response_residual"):
                high = 12
            if measurement is None:
                if template["illumination"][key] is not None:
                    raise ValueError("missing illumination measurement")
            elif not _number(measurement, low, high):
                raise ValueError("invalid illumination measurement")
    if illumination["historical"] != (illumination["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical illumination diagnostics")
    if illumination["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical illumination diagnostics")
    score = scores["illumination_consistency"]
    if score["available"] and (value["terminal"] or tracking["state"] != "STABLE" or not quality["acceptable"]
                               or illumination["historical"] or illumination["status"] != "observing"
                               or not illumination["face_regions_available"] or not illumination["ambient_available"]
                               or not illumination["pose_available"] or illumination["face_region_count"] != 7
                               or illumination["ambient_region_count"] != 4 or illumination["minimum_region_pixels"] < 24
                               or illumination["clipped_fraction"] > 0.10
                               or illumination["window_frame_count"] < 8 or illumination["window_duration_ms"] < 600):
        raise ValueError("illumination evidence outside valid observation")
    if illumination["ambient_available"]:
        balances = [illumination[key] for key in ("ambient_red_balance", "ambient_green_balance", "ambient_blue_balance")]
        if any(v is None for v in balances) or abs(sum(balances) - 1) > 1e-6 or illumination["ambient_luminance"] is None:
            raise ValueError("incomplete ambient photometry")
    if illumination["consistent_response_observed"] or score["available"] and score["confidence"] > 0.5:
        limits = (("ambient_spatial_change", 0.05, 12), ("face_spatial_change", 0.03, 12),
                  ("regional_response_residual", 0.012, 12), ("response_coherence", 0.80, 1),
                  ("direction_agreement", 0.80, 1), ("color_response_error", 0, 0.35),
                  ("head_rotation_span_degrees", 0, 1.5), ("face_motion_px", 0, 2), ("background_motion_px", 0, 2))
        if (not illumination["consistent_response_observed"] or illumination["motion_suppressed"]
                or illumination["response_mismatch"] or not illumination["motion_context_available"]
                or not illumination["face_regions_available"] or not illumination["ambient_available"]
                or not illumination["pose_available"] or illumination["window_frame_count"] < 8
                or illumination["window_duration_ms"] < 600 or illumination["changing_face_regions"] < 4
                or any(not _number(illumination[key], low, high) for key, low, high in limits)):
            raise ValueError("positive illumination evidence without diverse spatial response")
    replay = value["replay"]
    if not isinstance(replay, dict) or set(replay) != set(template["replay"]):
        raise ValueError("unknown replay fields")
    counter_limits = {"window_frame_count": 64, "window_duration_ms": 2000,
                      "face_region_count": 5, "background_region_count": 4,
                      "minimum_patch_width_px": 2000, "affected_face_regions": 5,
                      "artifact_family_count": 7}
    for key, measurement in replay.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid replay status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid replay event")
        elif isinstance(template["replay"][key], bool):
            if not isinstance(measurement, bool):
                raise ValueError("invalid replay state")
        elif key in counter_limits:
            if (isinstance(measurement, bool) or not isinstance(measurement, int)
                    or not 0 <= measurement <= counter_limits[key]):
                raise ValueError("invalid replay counter")
        else:
            high = 10000 if key.endswith("motion_px") else 1
            if measurement is None:
                if template["replay"][key] is not None:
                    raise ValueError("missing replay measurement")
            elif not _number(measurement, 0, high):
                raise ValueError("invalid replay measurement")
    if replay["historical"] != (replay["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical replay diagnostics")
    if replay["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical replay diagnostics")
    score = scores["replay_artifact_detection"]
    if score["available"] and (value["terminal"] or tracking["state"] != "STABLE" or not quality["acceptable"]
                               or replay["historical"] or replay["status"] != "observing"
                               or not replay["regions_available"] or not replay["background_available"]
                               or replay["face_region_count"] != 5 or replay["background_region_count"] != 4
                               or replay["minimum_patch_width_px"] < 17 or replay["texture_contrast"] is None
                               or replay["window_frame_count"] < 6 or replay["window_duration_ms"] < 450):
        raise ValueError("replay evidence outside valid observation")
    if replay["replay_artifact_observed"] or score["available"] and score["confidence"] < 0.5:
        required = ("periodic_peak_score", "chroma_grid_score", "banding_score", "sharpening_score",
                    "resampling_score", "glare_score", "flat_edge_score", "background_artifact_score",
                    "face_motion_px", "background_motion_px")
        if (not replay["replay_artifact_observed"] or replay["motion_suppressed"]
                or not replay["motion_context_available"] or replay["window_frame_count"] < 6
                or replay["window_duration_ms"] < 450 or replay["affected_face_regions"] < 3
                or replay["artifact_family_count"] < 2 or replay["artifact_persistence"] < .75
                or replay["background_artifact_score"] is None or replay["background_artifact_score"] > .45
                or replay["face_motion_px"] is None or replay["face_motion_px"] > 3
                or replay["background_motion_px"] is None or replay["background_motion_px"] > 3
                or any(replay[key] is None for key in required)):
            raise ValueError("replay artifact evidence without persistent multifamily response")
    temporal = value["temporal"]
    if not isinstance(temporal, dict) or set(temporal) != set(template["temporal"]):
        raise ValueError("unknown temporal fields")
    temporal_counters = {"window_frame_count": 64, "window_duration_ms": 2000,
                         "positive_group_count": 3, "supporting_signal_count": 8,
                         "conflict_count": 64, "missing_group_count": 3}
    temporal_floats = {"landmark_group_support", "photometric_group_support", "replay_suspicion",
                       "support_overlap", "cross_group_agreement", "temporal_stability",
                       "dependence_adjusted_support"}
    for key, measurement in temporal.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid temporal status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid temporal event")
        elif key in ("historical", "landmark_group_available", "photometric_group_available",
                     "replay_group_available", "motion_context_available", "conflict_observed",
                     "motion_suppressed", "consistent_behavior_observed"):
            if not isinstance(measurement, bool):
                raise ValueError("invalid temporal availability")
        elif key in temporal_counters:
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= temporal_counters[key]:
                raise ValueError("invalid temporal counter")
        elif key in temporal_floats:
            if not _number(measurement):
                raise ValueError("invalid temporal score")
        else:
            raise ValueError("unknown temporal measurement")
    if temporal["historical"] != (temporal["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical temporal diagnostics")
    if temporal["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical temporal diagnostics")
    temporal_score = scores["temporal_consistency"]
    if temporal_score["available"] and (value["terminal"] or tracking["state"] != "STABLE"
                                         or not quality["acceptable"] or temporal["historical"]
                                         or temporal["status"] != "observing" or temporal["window_frame_count"] < 8
                                         or temporal["window_duration_ms"] < 600):
        raise ValueError("temporal evidence outside valid observation")
    if temporal_score["available"] and temporal_score["confidence"] > 0.5:
        if (temporal["positive_group_count"] < 2 or temporal["support_overlap"] < 0.5
                or temporal["cross_group_agreement"] < 0.65 or temporal["conflict_observed"]
                or temporal["motion_suppressed"] or not temporal["motion_context_available"]
                or not temporal["consistent_behavior_observed"]):
            raise ValueError("positive temporal evidence without independent groups")
    fusion = value["fusion"]
    if not isinstance(fusion, dict) or set(fusion) != set(template["fusion"]):
        raise ValueError("unknown fusion fields")
    fusion_counters = {"window_frame_count": 64, "window_duration_ms": 2000,
                       "active_group_count": 3, "available_group_count": 3,
                       "missing_group_count": 3, "positive_group_count": 3,
                       "suspicious_group_count": 1, "conflict_count": 64,
                       "individual_signal_count": 8}
    fusion_floats = {"landmark_group_score", "photometric_group_score", "replay_group_score",
                     "dependence_penalty", "raw_combined_score", "adjusted_combined_score",
                     "coverage", "group_agreement"}
    for key, measurement in fusion.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid fusion status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid fusion event")
        elif key in ("historical", "liveness_candidate", "spoof_candidate"):
            if not isinstance(measurement, bool):
                raise ValueError("invalid fusion state")
        elif key in fusion_counters:
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= fusion_counters[key]:
                raise ValueError("invalid fusion counter")
        elif key in fusion_floats:
            if not _number(measurement):
                raise ValueError("invalid fusion score")
        else:
            raise ValueError("unknown fusion measurement")
    if fusion["historical"] != (fusion["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical fusion diagnostics")
    if fusion["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical fusion diagnostics")
    if fusion["liveness_candidate"] or fusion["spoof_candidate"]:
        raise ValueError("fusion candidates are disabled")
    observation = value["observation"]
    if not isinstance(observation, dict) or set(observation) != set(template["observation"]):
        raise ValueError("unknown observation fields")
    observation_counters = {"elapsed_ms": 60_000, "minimum_duration_ms": 30_000,
                            "base_limit_ms": 30_000, "effective_limit_ms": 35_000,
                            "extension_limit_ms": 5_000, "extension_count": 1,
                            "quality_frame_count": MAX_CAPTURE_FRAMES, "stable_frame_count": MAX_CAPTURE_FRAMES,
                            "positive_group_count": 3, "conflict_count": 64}
    for key, measurement in observation.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "extension_pending",
                                   "early_complete", "hard_timeout", "historical_diagnostics"):
                raise ValueError("invalid observation status")
        elif key == "event":
            if not _code(measurement):
                raise ValueError("invalid observation event")
        elif key == "challenge_state":
            if measurement != "disabled":
                raise ValueError("challenge policy is disabled")
        elif key in ("historical", "temporal_ready", "early_completion", "extension_requested"):
            if not isinstance(measurement, bool):
                raise ValueError("invalid observation state")
        elif key in observation_counters:
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= observation_counters[key]:
                raise ValueError("invalid observation counter")
        elif key == "quality_frame_fraction":
            if not _number(measurement):
                raise ValueError("invalid observation quality fraction")
        else:
            raise ValueError("unknown observation measurement")
    if observation["historical"] != (observation["status"] == "historical_diagnostics"):
        raise ValueError("inconsistent historical observation diagnostics")
    if observation["historical"] and (not value["terminal"] or tracking["state"] != "CLOSED"):
        raise ValueError("invalid historical observation diagnostics")
    if observation["quality_frame_count"] > value["frame_count"]:
        raise ValueError("observation quality count exceeds frame count")
    if observation["stable_frame_count"] > observation["quality_frame_count"]:
        raise ValueError("observation stable count exceeds quality count")
    if observation["effective_limit_ms"] < observation["base_limit_ms"]:
        raise ValueError("observation limit moved backwards")
    if (observation["minimum_duration_ms"] > observation["base_limit_ms"]
            or observation["effective_limit_ms"] != observation["base_limit_ms"] + observation["extension_count"] * observation["extension_limit_ms"]):
        raise ValueError("inconsistent observation duration limits")
    if observation["extension_requested"] != (observation["status"] == "extension_pending"):
        raise ValueError("inconsistent observation extension state")
    if observation["early_completion"] != (observation["status"] == "early_complete"):
        raise ValueError("inconsistent observation completion state")
    return value


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def supervise(command, *, duration, on_update=lambda result, preview: None,
              pump=lambda: True, startup_timeout=20.0, frame_timeout=2.0,
              extension_duration=0.0):
    """Private worker diagnostics only; never an authentication IPC endpoint."""
    process = None
    selector = selectors.DefaultSelector()
    try:
        if not _number(duration, 0.5, 30) or not _number(extension_duration, 0, 5):
            return failure_result("invalid_observation_limits")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        created = last_update = time.monotonic()
        started = None
        final = None
        buffer = bytearray()
        while True:
            if not pump():
                return failure_result("cancelled")
            now = time.monotonic()
            if started is None and now - created > startup_timeout:
                return failure_result("worker_startup_timeout")
            if started is not None:
                if now - started > duration + extension_duration + 0.5:
                    return failure_result("worker_observation_timeout")
                if final is None and now - last_update > frame_timeout:
                    return failure_result("camera_or_model_stalled")
            if not selector.select(timeout=0.05):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                if buffer or final is None:
                    return failure_result("worker_exited_without_result")
                try:
                    exit_code = process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    return failure_result("worker_cleanup_stalled")
                return final if exit_code == 0 else failure_result("worker_crashed")
            buffer.extend(chunk)
            if len(buffer) > MAX_MESSAGE_BYTES:
                return failure_result("oversized_worker_message")
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                message = json.loads(line)
                if not isinstance(message, dict) or final is not None:
                    return failure_result("invalid_worker_protocol")
                kind = message.get("kind")
                if kind == "started" and set(message) == {"kind"} and started is None:
                    started = last_update = time.monotonic()
                elif kind in ("update", "final"):
                    if set(message) not in ({"kind", "result"}, {"kind", "result", "preview"}):
                        return failure_result("invalid_worker_protocol")
                    result = validate_result(message["result"])
                    if kind == "final":
                        if not result["terminal"] or "preview" in message:
                            return failure_result("invalid_worker_protocol")
                        final = result
                    else:
                        if started is None:
                            return failure_result("invalid_worker_protocol")
                        preview = message.get("preview")
                        if preview is not None and (not isinstance(preview, str) or len(preview) > 1_800_000):
                            return failure_result("invalid_preview")
                        last_update = time.monotonic()
                        on_update(result, preview)
                else:
                    return failure_result("invalid_worker_protocol")
    except KeyboardInterrupt:
        return failure_result("cancelled")
    except Exception:
        return failure_result("harness_or_ipc_error")
    finally:
        selector.close()
        if process is not None:
            stop_worker(process)
            process.stdout.close()

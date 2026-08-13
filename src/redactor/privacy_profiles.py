from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from redactor import detection, plate_detection, plate_tracking, privacy_gap_recovery, rendering
from redactor.privacy_records import CATEGORY_FACE, CATEGORY_PLATE, stable_hash

SCHEMA_VERSION = "privacy_category_profile.v1"


@dataclass(frozen=True)
class CategoryProfile:
    category: str
    detector_id: str
    detector_configuration_hash: str
    tracking_profile_id: str
    tracking_profile_hash: str
    render_profile_id: str
    render_profile_hash: str
    geometry: dict[str, Any] = field(default_factory=dict)
    association: dict[str, Any] = field(default_factory=dict)
    confirmation: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    gap_recovery: dict[str, Any] = field(default_factory=dict)
    rendering: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "association": self.association,
            "category": self.category,
            "confirmation": self.confirmation,
            "detector_configuration_hash": self.detector_configuration_hash,
            "detector_id": self.detector_id,
            "evidence": self.evidence,
            "gap_recovery": self.gap_recovery,
            "geometry": self.geometry,
            "render_profile_hash": self.render_profile_hash,
            "render_profile_id": self.render_profile_id,
            "rendering": self.rendering,
            "review": self.review,
            "schema_version": self.schema_version,
            "tracking_profile_hash": self.tracking_profile_hash,
            "tracking_profile_id": self.tracking_profile_id,
        }

    def profile_hash(self) -> str:
        return stable_hash(self.to_dict())


def face_profile() -> CategoryProfile:
    recovery = privacy_gap_recovery.PrivacyGapRecoverySettings()
    render_settings = rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
        preview_watermark=False,
    )
    render_payload = {
        "blur_kernel": render_settings.normalized_blur_kernel,
        "margin_fraction": render_settings.margin_fraction,
        "margin_pixels": render_settings.margin_pixels,
        "method": render_settings.method,
        "pixel_size": render_settings.pixel_size,
    }
    tracking_payload = {
        "association_center_distance": 0.75,
        "association_iou_threshold": 0.15,
        "confirmation_window": 5,
        "max_gap_seconds": 0.30,
        "min_observed_hits": 3,
        "min_strong_hits": 2,
        "strong_threshold": 0.30,
        "weak_threshold": 0.20,
    }
    recovery_payload = privacy_gap_recovery.settings_to_dict(recovery)
    evidence_payload = {
        "active_in_production": False,
        "continuation_threshold": 0.55,
        "confirmation_threshold": 0.70,
        "max_safe_gap_seconds": 0.30,
        "min_mean_confidence": 0.30,
        "min_observation_count": 3,
        "min_observation_density": 0.25,
        "profile_id": "face_temporal_evidence_compatibility_v1",
        "reason": "production face tracking remains on the frozen v1 tracker/recovery path",
    }
    return CategoryProfile(
        category=CATEGORY_FACE,
        detector_id=detection.DETECTOR_BACKEND_ADAS,
        detector_configuration_hash="openvino_face_adas_0001_threshold_0.20_0.30",
        tracking_profile_id="face_tracking_v1_adas",
        tracking_profile_hash=stable_hash(tracking_payload),
        render_profile_id="face_pixelate_preview_v1",
        render_profile_hash=stable_hash(render_payload),
        geometry={
            "minimum_area_fraction": 0.0005,
            "maximum_area_fraction": 0.08,
            "box_shape": "axis_aligned_face_box",
        },
        association={
            "association_center_distance": tracking_payload["association_center_distance"],
            "association_iou_threshold": tracking_payload["association_iou_threshold"],
        },
        confirmation={
            "confirmation_window": tracking_payload["confirmation_window"],
            "min_observed_hits": tracking_payload["min_observed_hits"],
            "min_strong_hits": tracking_payload["min_strong_hits"],
        },
        evidence=evidence_payload,
        gap_recovery=recovery_payload,
        rendering=render_payload,
        review={
            "final_blocking_statuses": sorted(rendering.FINAL_BLOCKING_STATUSES),
            "default_issue_category": "privacy_gap_or_identity_risk",
        },
    )


def plate_profile() -> CategoryProfile:
    tracking_config = plate_tracking.PlateTrackingConfig()
    evidence_config = plate_tracking.default_evidence_config()
    render_payload = {
        "blur_kernel": 51,
        "margin_fraction": 0.15,
        "margin_pixels": 0,
        "method": rendering.METHOD_PIXELATE,
        "pixel_size": 16,
    }
    return CategoryProfile(
        category=CATEGORY_PLATE,
        detector_id=plate_detection.DETECTOR_ID,
        detector_configuration_hash=plate_detection.OIMPlateDetectorConfig().configuration_hash(),
        tracking_profile_id="plate_tracking_v1",
        tracking_profile_hash=tracking_config.configuration_hash(),
        render_profile_id="plate_pixelate_preview_v1",
        render_profile_hash=stable_hash(render_payload),
        geometry={
            "maximum_area_fraction": tracking_config.max_area_fraction,
            "maximum_aspect_ratio": tracking_config.max_aspect_ratio,
            "minimum_area_fraction": tracking_config.min_area_fraction,
            "minimum_aspect_ratio": tracking_config.min_aspect_ratio,
        },
        association={
            "max_association_gap_seconds": tracking_config.max_association_gap_seconds,
            "max_association_scale_ratio": tracking_config.max_association_scale_ratio,
            "max_center_distance_norm": tracking_config.max_center_distance_norm,
            "min_association_iou": tracking_config.min_association_iou,
        },
        confirmation={
            "min_accepted_observation_density": tracking_config.min_accepted_observation_density,
            "min_accepted_observations": tracking_config.min_accepted_observations,
            "min_review_observations": tracking_config.min_review_observations,
        },
        evidence={
            "active_in_production": True,
            "configuration": evidence_config.to_dict(),
            "configuration_hash": evidence_config.configuration_hash(),
        },
        gap_recovery={
            "max_safe_interpolation_gap_seconds": (
                tracking_config.max_safe_interpolation_gap_seconds
            ),
            "scene_cut_luma_delta": tracking_config.scene_cut_luma_delta,
        },
        rendering=render_payload,
        review={
            "accepted_track_actions": ["Correct", "Needs correction", "Not sensitive"],
            "unresolved_gap_actions": ["Correct", "Needs correction", "Not sensitive"],
        },
    )


def profile_for(category: str) -> CategoryProfile:
    if category == CATEGORY_FACE:
        return face_profile()
    if category == CATEGORY_PLATE:
        return plate_profile()
    raise ValueError(f"unsupported privacy category profile: {category}")


def profile_payload_hash(payload: dict[str, Any]) -> str:
    return stable_hash(json.loads(json.dumps(payload, sort_keys=True)))

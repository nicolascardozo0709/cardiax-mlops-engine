"""Unit tests for CardiaX morphological and temporal feature extraction."""

import numpy as np
import pytest
from src.data.synthetic_generator import SyntheticECGGenerator
from src.features.wave_morphology import HeartbeatFeatureExtractor, FEATURE_NAMES


def test_extract_single_beat_features_completeness():
    """Verify feature extractor generates all defined features without NaNs or Infs."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    beat, _ = generator.generate_single_beat("N", window_duration_s=0.8)

    extractor = HeartbeatFeatureExtractor(sampling_rate_hz=360)
    feats = extractor.extract_single_beat_features(beat, pre_rr_s=0.82, post_rr_s=0.81)

    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature {name}"
        assert not np.isnan(feats[name]), f"Feature {name} is NaN"
        assert not np.isinf(feats[name]), f"Feature {name} is Inf"


def test_qrs_duration_difference_between_normal_and_pvc():
    """Verify Ventricular Ectopic Beats (PVC) exhibit wider QRS duration than Normal beats."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    normal_beat, _ = generator.generate_single_beat("N", window_duration_s=0.8)
    pvc_beat, _ = generator.generate_single_beat("V", window_duration_s=0.8)

    extractor = HeartbeatFeatureExtractor(sampling_rate_hz=360)
    feats_n = extractor.extract_single_beat_features(normal_beat)
    feats_v = extractor.extract_single_beat_features(pvc_beat)

    # PVC has wider QRS duration and higher energy in general
    assert feats_v["qrs_duration_ms"] > feats_n["qrs_duration_ms"]
    assert feats_v["t_wave_polarity"] == -1.0  # Discordant T-wave inversion in PVC

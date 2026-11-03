"""Unit tests for CardiaX Digital Signal Processing and Pan-Tompkins QRS detector."""

import numpy as np
import pytest
from src.features.signal_filter import ECGFilter, PanTompkinsQRSDetector
from src.data.synthetic_generator import SyntheticECGGenerator


def test_butter_bandpass_attenuates_dc_and_high_frequencies():
    """Verify Butterworth bandpass eliminates DC offset and high-frequency noise."""
    fs = 360
    ecg_filter = ECGFilter(sampling_rate_hz=fs)
    t = np.arange(fs * 2) / fs

    # Signal with DC offset (2.0) and high frequency noise (100 Hz)
    raw = 2.0 + np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 100.0 * t)
    filtered = ecg_filter.clean_signal(raw)

    # Mean after DC blocking should be near 0
    assert abs(np.mean(filtered)) < 0.15
    # High frequency 100 Hz power should be attenuated
    assert np.max(filtered) < np.max(raw)


def test_pan_tompkins_peak_detection_count():
    """Verify Pan-Tompkins detects the expected number of heartbeats in a rhythm strip."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    # Generate 5-second normal rhythm at 72 BPM (~6 beats)
    strip = generator.generate_rhythm_strip(duration_s=5.0, heart_rate_bpm=72.0, arrhythmia_pattern="normal")

    ecg_filter = ECGFilter(sampling_rate_hz=360)
    cleaned = ecg_filter.clean_signal(strip["raw_signal"])

    detector = PanTompkinsQRSDetector(sampling_rate_hz=360)
    detected_peaks = detector.detect_peaks(cleaned)

    # Should detect 5 to 7 beats
    assert 5 <= len(detected_peaks) <= 7

    # Ensure all inter-beat intervals respect refractory period (> 200 ms)
    diffs = np.diff(detected_peaks)
    assert np.all(diffs >= int(0.200 * 360))

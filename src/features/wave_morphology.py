"""CardiaX Morphological & Clinical Feature Extraction Module.

Extracts cardiac features from isolated ECG heartbeats and rhythm strips:
- QRS complex width, amplitude, energy, and area
- P-wave and T-wave presence, amplitude, and polarity
- ST-segment deviation
- Temporal intervals (Pre-RR, Post-RR, RR ratio)
- Statistical moments (skewness, kurtosis, variance)
- Spectral power distribution and spectral entropy
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import scipy.stats as stats


FEATURE_NAMES = [
    "qrs_amplitude",
    "qrs_duration_ms",
    "qrs_energy",
    "qrs_area",
    "signal_variance",
    "signal_skewness",
    "signal_kurtosis",
    "p_wave_amplitude",
    "t_wave_amplitude",
    "t_wave_polarity",
    "st_elevation_mv",
    "pt_amplitude_ratio",
    "power_low_band",
    "power_qrs_band",
    "power_high_band",
    "spectral_entropy",
    "pre_rr_ratio",
    "post_rr_ratio",
    "rr_imbalance",
]


class HeartbeatFeatureExtractor:
    """Extracts clinical feature vectors from single-beat ECG arrays and rhythm contexts."""

    def __init__(self, sampling_rate_hz: int = 360):
        self.fs = sampling_rate_hz

    def extract_single_beat_features(
        self,
        beat_waveform: np.ndarray,
        pre_rr_s: Optional[float] = None,
        post_rr_s: Optional[float] = None,
        nominal_rr_s: float = 0.833,  # ~72 BPM default
    ) -> Dict[str, float]:
        """Compute the full clinical feature dictionary for an isolated heartbeat.

        Args:
            beat_waveform: 1D array of single beat centered near R-peak (e.g. 288 samples for 0.8s @ 360Hz).
            pre_rr_s: Time in seconds from previous R-peak (if available in rhythm context).
            post_rr_s: Time in seconds to next R-peak (if available in rhythm context).
            nominal_rr_s: Baseline nominal RR interval for normalization.

        Returns:
            Dictionary mapping feature names to numerical values.
        """
        w = np.asarray(beat_waveform, dtype=float)
        n = len(w)
        center_idx = n // 2

        # 1. Morphological amplitude and power
        r_peak_val = float(w[center_idx])
        min_val = float(np.min(w))
        max_val = float(np.max(w))
        qrs_amplitude = max_val - min_val

        # QRS Window: [-60ms, +60ms] around center
        qrs_half_samples = int(0.060 * self.fs)
        qrs_start = max(0, center_idx - qrs_half_samples)
        qrs_end = min(n, center_idx + qrs_half_samples)
        qrs_segment = w[qrs_start:qrs_end]

        qrs_energy = float(np.sum(qrs_segment ** 2))
        qrs_area = float(np.sum(np.abs(qrs_segment)))

        # QRS duration estimation (samples where power > 15% of peak)
        threshold = 0.15 * max(1e-4, np.max(np.abs(qrs_segment)))
        above_thresh = np.where(np.abs(qrs_segment) > threshold)[0]
        if len(above_thresh) > 1:
            qrs_duration_ms = float((above_thresh[-1] - above_thresh[0]) / self.fs * 1000.0)
        else:
            qrs_duration_ms = 40.0

        # 2. Statistical Moments
        var_val = float(np.var(w))
        skew_val = float(stats.skew(w)) if var_val > 1e-8 else 0.0
        kurt_val = float(stats.kurtosis(w)) if var_val > 1e-8 else 0.0

        # 3. P-wave and T-wave features
        # P-wave: [-220ms, -80ms]
        p_start = max(0, center_idx - int(0.220 * self.fs))
        p_end = max(0, center_idx - int(0.080 * self.fs))
        p_segment = w[p_start:p_end] if p_end > p_start else np.array([0.0])
        p_amp = float(np.max(p_segment) if len(p_segment) > 0 else 0.0)

        # ST segment: [+60ms, +120ms]
        st_start = min(n, center_idx + int(0.060 * self.fs))
        st_end = min(n, center_idx + int(0.120 * self.fs))
        st_segment = w[st_start:st_end] if st_end > st_start else np.array([0.0])
        st_elevation = float(np.mean(st_segment) if len(st_segment) > 0 else 0.0)

        # T-wave: [+140ms, +320ms]
        t_start = min(n, center_idx + int(0.140 * self.fs))
        t_end = min(n, center_idx + int(0.320 * self.fs))
        t_segment = w[t_start:t_end] if t_end > t_start else np.array([0.0])
        if len(t_segment) > 0:
            t_max = np.max(t_segment)
            t_min = np.min(t_segment)
            # Pick extreme deviation
            t_amp = float(t_max if abs(t_max) >= abs(t_min) else t_min)
            t_pol = 1.0 if t_amp >= 0 else -1.0
        else:
            t_amp = 0.0
            t_pol = 1.0

        pt_ratio = float(p_amp / max(1e-4, abs(t_amp)))

        # 4. Spectral Features via FFT
        fft_vals = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        total_power = np.sum(fft_vals ** 2) + 1e-9

        # Band powers: Low (0-5Hz), QRS (5-15Hz), High (15-40Hz)
        mask_low = (freqs >= 0.0) & (freqs < 5.0)
        mask_qrs = (freqs >= 5.0) & (freqs < 15.0)
        mask_high = (freqs >= 15.0) & (freqs <= 40.0)

        pow_low = float(np.sum(fft_vals[mask_low] ** 2) / total_power)
        pow_qrs = float(np.sum(fft_vals[mask_qrs] ** 2) / total_power)
        pow_high = float(np.sum(fft_vals[mask_high] ** 2) / total_power)

        # Spectral entropy: H = - sum(p * log2(p))
        prob_dist = (fft_vals ** 2) / total_power
        prob_dist = prob_dist[prob_dist > 1e-12]
        spec_entropy = float(-np.sum(prob_dist * np.log2(prob_dist)))

        # 5. Timing intervals (relative to nominal baseline)
        if pre_rr_s is None:
            pre_rr_ratio = 1.0
        else:
            pre_rr_ratio = float(pre_rr_s / max(0.1, nominal_rr_s))

        if post_rr_s is None:
            post_rr_ratio = 1.0
        else:
            post_rr_ratio = float(post_rr_s / max(0.1, nominal_rr_s))

        rr_imbalance = float(pre_rr_ratio / max(0.1, post_rr_ratio))

        features: Dict[str, float] = {
            "qrs_amplitude": qrs_amplitude,
            "qrs_duration_ms": qrs_duration_ms,
            "qrs_energy": qrs_energy,
            "qrs_area": qrs_area,
            "signal_variance": var_val,
            "signal_skewness": skew_val,
            "signal_kurtosis": kurt_val,
            "p_wave_amplitude": p_amp,
            "t_wave_amplitude": t_amp,
            "t_wave_polarity": t_pol,
            "st_elevation_mv": st_elevation,
            "pt_amplitude_ratio": pt_ratio,
            "power_low_band": pow_low,
            "power_qrs_band": pow_qrs,
            "power_high_band": pow_high,
            "spectral_entropy": spec_entropy,
            "pre_rr_ratio": pre_rr_ratio,
            "post_rr_ratio": post_rr_ratio,
            "rr_imbalance": rr_imbalance,
        }
        return features

    def extract_features_matrix(
        self,
        waveforms: np.ndarray,
        pre_rr_list: Optional[List[float]] = None,
        post_rr_list: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Extract a 2D numpy feature matrix from multiple beat waveforms.

        Args:
            waveforms: 2D array of shape [num_beats, samples_per_beat].
            pre_rr_list: Optional list of pre-RR values for each beat.
            post_rr_list: Optional list of post-RR values for each beat.

        Returns:
            2D numpy array of shape [num_beats, num_features].
        """
        rows: List[List[float]] = []
        num_beats = len(waveforms)

        for i in range(num_beats):
            pre_val = pre_rr_list[i] if pre_rr_list is not None else None
            post_val = post_rr_list[i] if post_rr_list is not None else None
            feat_dict = self.extract_single_beat_features(waveforms[i], pre_rr_s=pre_val, post_rr_s=post_val)
            row = [feat_dict[k] for k in FEATURE_NAMES]
            rows.append(row)

        return np.array(rows, dtype=float)

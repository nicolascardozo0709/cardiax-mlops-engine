"""CardiaX Synthetic Electrocardiogram Generator.

Implements a physiologically grounded dynamical model for Lead II ECG synthesis,
generating realistic multi-class cardiac arrhythmias under the ANSI/AAMI EC57 standard:
- Class N: Normal sinus beat
- Class S: Supraventricular ectopic beat (PAC / SVEB)
- Class V: Premature ventricular contraction (PVC / VEB)
- Class F: Fusion beat (simultaneous supraventricular + ventricular activation)
- Class Q: Paced / unclassifiable beat with pacemaker spike
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np


class SyntheticECGGenerator:
    """Generates authentic Lead II ECG signals and individual heartbeat waveforms

    simulating AAMI EC57 classes with physiologically accurate P-QRS-T morphology.
    """

    AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
    AAMI_NAMES = {
        "N": "Normal Sinus Beat",
        "S": "Supraventricular Ectopic Beat (SVEB)",
        "V": "Ventricular Ectopic Beat (VEB / PVC)",
        "F": "Fusion Beat",
        "Q": "Paced / Unclassified Beat",
    }

    def __init__(self, sampling_rate_hz: int = 360, random_state: int | None = 42):
        """Initialize the synthetic ECG generator.

        Args:
            sampling_rate_hz: Sampling frequency in Hertz (default: 360 Hz, matching MIT-BIH).
            random_state: Seed for reproducibility.
        """
        self.fs = sampling_rate_hz
        self.rng = np.random.default_rng(random_state)

    def _generate_gaussian_wave(
        self, t: np.ndarray, amplitude: float, center: float, width: float
    ) -> np.ndarray:
        """Helper to generate a Gaussian deflection representing a cardiac wave component."""
        return amplitude * np.exp(-0.5 * ((t - center) / max(width, 1e-5)) ** 2)

    def generate_single_beat(
        self,
        beat_type: str = "N",
        window_duration_s: float = 0.8,
        noise_level: float = 0.02,
    ) -> Tuple[np.ndarray, str]:
        """Synthesize an isolated single heartbeat centered at R-peak.

        Args:
            beat_type: One of 'N', 'S', 'V', 'F', 'Q'.
            window_duration_s: Total window width around the beat (default 0.8s).
            noise_level: Standard deviation of additive Gaussian baseline noise.

        Returns:
            Tuple of (waveform array of shape [num_samples], beat_type).
        """
        if beat_type not in self.AAMI_CLASSES:
            raise ValueError(f"Unknown beat_type '{beat_type}'. Must be one of {self.AAMI_CLASSES}")

        num_samples = int(window_duration_s * self.fs)
        # Center R-peak at t = 0
        t = np.linspace(-window_duration_s / 2, window_duration_s / 2, num_samples)
        signal = np.zeros_like(t)

        if beat_type == "N":
            # Normal beat: Clean P, sharp tall QRS, normal T
            p = self._generate_gaussian_wave(t, amplitude=0.18, center=-0.16, width=0.035)
            q = self._generate_gaussian_wave(t, amplitude=-0.12, center=-0.03, width=0.012)
            r = self._generate_gaussian_wave(t, amplitude=1.20, center=0.0, width=0.020)
            s = self._generate_gaussian_wave(t, amplitude=-0.25, center=0.035, width=0.016)
            t_wave = self._generate_gaussian_wave(t, amplitude=0.28, center=0.22, width=0.065)
            signal = p + q + r + s + t_wave

        elif beat_type == "S":
            # Supraventricular Ectopic Beat (PAC):
            # Narrow QRS (His-Purkinje intact), but distorted/hidden P-wave and slightly modified T
            p = self._generate_gaussian_wave(t, amplitude=0.08, center=-0.12, width=0.025)
            q = self._generate_gaussian_wave(t, amplitude=-0.08, center=-0.025, width=0.012)
            r = self._generate_gaussian_wave(t, amplitude=1.05, center=0.0, width=0.018)
            s = self._generate_gaussian_wave(t, amplitude=-0.20, center=0.03, width=0.015)
            t_wave = self._generate_gaussian_wave(t, amplitude=0.22, center=0.18, width=0.055)
            signal = p + q + r + s + t_wave

        elif beat_type == "V":
            # Ventricular Ectopic Beat (PVC):
            # Wide, notched QRS (>120ms), absent P wave, discordant inverted T wave
            r_wide = self._generate_gaussian_wave(t, amplitude=1.55, center=-0.02, width=0.055)
            r_notch = self._generate_gaussian_wave(t, amplitude=0.45, center=0.03, width=0.040)
            s_deep = self._generate_gaussian_wave(t, amplitude=-0.65, center=0.07, width=0.045)
            t_inverted = self._generate_gaussian_wave(t, amplitude=-0.42, center=0.26, width=0.080)
            signal = r_wide + r_notch + s_deep + t_inverted

        elif beat_type == "F":
            # Fusion Beat: Intermediate between normal and ventricular morphology
            p_small = self._generate_gaussian_wave(t, amplitude=0.10, center=-0.15, width=0.030)
            q = self._generate_gaussian_wave(t, amplitude=-0.15, center=-0.04, width=0.018)
            r = self._generate_gaussian_wave(t, amplitude=1.35, center=0.0, width=0.035)
            s = self._generate_gaussian_wave(t, amplitude=-0.40, center=0.05, width=0.025)
            t_wave = self._generate_gaussian_wave(t, amplitude=0.10, center=0.24, width=0.070)
            signal = p_small + q + r + s + t_wave

        elif beat_type == "Q":
            # Paced / Unknown Beat: Sharp spike (pacer impulse) followed by wide depolarization
            spike = self._generate_gaussian_wave(t, amplitude=1.80, center=-0.05, width=0.003)
            r_paced = self._generate_gaussian_wave(t, amplitude=0.90, center=0.02, width=0.045)
            s_paced = self._generate_gaussian_wave(t, amplitude=-0.50, center=0.08, width=0.035)
            t_paced = self._generate_gaussian_wave(t, amplitude=-0.30, center=0.25, width=0.070)
            signal = spike + r_paced + s_paced + t_paced

        # Add physiological white noise
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, size=num_samples)
            signal += noise

        return signal, beat_type

    def generate_rhythm_strip(
        self,
        duration_s: float = 10.0,
        heart_rate_bpm: float = 72.0,
        arrhythmia_pattern: str = "normal",
        noise_level: float = 0.03,
    ) -> Dict[str, any]:
        """Generate a continuous ECG lead II rhythm strip containing annotated heartbeats.

        Args:
            duration_s: Total strip length in seconds.
            heart_rate_bpm: Nominal baseline heart rate in beats per minute.
            arrhythmia_pattern: One of:
                - 'normal': Pure sinus rhythm.
                - 'bigeminy': Alternating Normal and Ventricular Ectopic beats (N-V-N-V).
                - 'trigeminy': Two Normal beats followed by one Ventricular Ectopic beat (N-N-V).
                - 'couplet': Normal rhythm with occasional paired PVCs.
                - 'svt': Episodes of supraventricular premature contractions.
                - 'mixed': Clinical mixture of N, S, V, F, Q beats.
            noise_level: Background EMG and baseline wander intensity.

        Returns:
            Dictionary containing:
                - 'time': 1D array of timestamps (s)
                - 'signal': 1D array of raw ECG amplitude (mV)
                - 'r_peaks': List of integer sample indices of R-peaks
                - 'annotations': List of AAMI class labels ('N', 'S', 'V', 'F', 'Q')
                - 'heart_rate_bpm': Actual mean heart rate
                - 'sampling_rate_hz': 360
        """
        nominal_rr_s = 60.0 / heart_rate_bpm
        total_samples = int(duration_s * self.fs)
        time_arr = np.arange(total_samples) / self.fs
        signal = np.zeros(total_samples)

        # Generate beat schedule
        r_peaks: List[int] = []
        annotations: List[str] = []

        current_time_s = 0.4  # Start after brief lead-in
        beat_idx = 0

        while current_time_s < duration_s - 0.4:
            # Determine beat type and next RR interval based on pattern
            if arrhythmia_pattern == "normal":
                b_type = "N"
                next_rr = nominal_rr_s * self.rng.normal(1.0, 0.03)

            elif arrhythmia_pattern == "bigeminy":
                if beat_idx % 2 == 1:
                    b_type = "V"
                    # Compensatory pause after PVC
                    next_rr = nominal_rr_s * 1.35
                else:
                    b_type = "N"
                    next_rr = nominal_rr_s * 0.75  # Premature coupling

            elif arrhythmia_pattern == "trigeminy":
                if beat_idx % 3 == 2:
                    b_type = "V"
                    next_rr = nominal_rr_s * 1.30
                else:
                    b_type = "N"
                    next_rr = nominal_rr_s * (0.85 if beat_idx % 3 == 1 else 1.0)

            elif arrhythmia_pattern == "couplet":
                if beat_idx in [4, 5, 12, 13]:
                    b_type = "V"
                    next_rr = nominal_rr_s * (0.70 if beat_idx in [4, 12] else 1.40)
                else:
                    b_type = "N"
                    next_rr = nominal_rr_s * self.rng.normal(1.0, 0.03)

            elif arrhythmia_pattern == "svt":
                if beat_idx in [3, 7, 11, 15]:
                    b_type = "S"
                    next_rr = nominal_rr_s * 0.72  # Early PAC
                else:
                    b_type = "N"
                    next_rr = nominal_rr_s * self.rng.normal(1.0, 0.02)

            else:  # 'mixed'
                rand_val = self.rng.random()
                if rand_val < 0.65:
                    b_type = "N"
                    next_rr = nominal_rr_s * self.rng.normal(1.0, 0.03)
                elif rand_val < 0.80:
                    b_type = "V"
                    next_rr = nominal_rr_s * 1.30
                elif rand_val < 0.90:
                    b_type = "S"
                    next_rr = nominal_rr_s * 0.75
                elif rand_val < 0.96:
                    b_type = "F"
                    next_rr = nominal_rr_s * 1.05
                else:
                    b_type = "Q"
                    next_rr = nominal_rr_s * 1.00

            peak_sample = int(round(current_time_s * self.fs))
            r_peaks.append(peak_sample)
            annotations.append(b_type)

            # Synthesize single beat waveform
            beat_wf, _ = self.generate_single_beat(b_type, window_duration_s=0.7, noise_level=0.0)
            half_win = len(beat_wf) // 2

            # Place into continuous strip
            start_s = peak_sample - half_win
            end_s = peak_sample + (len(beat_wf) - half_win)

            src_start = 0
            src_end = len(beat_wf)

            if start_s < 0:
                src_start = -start_s
                start_s = 0
            if end_s > total_samples:
                src_end = len(beat_wf) - (end_s - total_samples)
                end_s = total_samples

            signal[start_s:end_s] += beat_wf[src_start:src_end]

            current_time_s += next_rr
            beat_idx += 1

        # Add realistic physiological artifacts
        # 1. Baseline wander (respiration ~0.2 Hz)
        baseline_wander = 0.12 * np.sin(2 * np.pi * 0.22 * time_arr)
        # 2. Powerline interference (60 Hz hum)
        powerline_hum = 0.015 * np.sin(2 * np.pi * 60.0 * time_arr)
        # 3. High-frequency muscle tremor (EMG white noise)
        emg_noise = self.rng.normal(0, noise_level, size=total_samples)

        clean_signal = signal.copy()
        noisy_signal = signal + baseline_wander + powerline_hum + emg_noise

        actual_bpm = (len(r_peaks) / (duration_s if duration_s > 0 else 1)) * 60.0

        return {
            "time": time_arr,
            "raw_signal": noisy_signal,
            "clean_reference": clean_signal,
            "r_peaks": r_peaks,
            "annotations": annotations,
            "heart_rate_bpm": float(round(actual_bpm, 1)),
            "sampling_rate_hz": self.fs,
            "arrhythmia_pattern": arrhythmia_pattern,
        }

    def generate_balanced_dataset(
        self, samples_per_class: int = 400
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate a balanced labeled dataset of single beat waveforms for ML training.

        Args:
            samples_per_class: Number of heartbeats per AAMI class.

        Returns:
            Tuple of (X: 2D array of shape [num_beats, samples_per_beat], y: List of labels).
        """
        all_waveforms: List[np.ndarray] = []
        all_labels: List[str] = []

        for b_type in self.AAMI_CLASSES:
            for _ in range(samples_per_class):
                # Variate window and noise slightly for generalization
                noise = self.rng.uniform(0.01, 0.04)
                wf, lbl = self.generate_single_beat(b_type, window_duration_s=0.8, noise_level=noise)
                all_waveforms.append(wf)
                all_labels.append(lbl)

        # Shuffle deterministically
        indices = np.arange(len(all_labels))
        self.rng.shuffle(indices)

        X = np.array([all_waveforms[i] for i in indices])
        y = [all_labels[i] for i in indices]

        return X, y

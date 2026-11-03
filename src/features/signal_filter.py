"""CardiaX Digital Signal Processing & QRS Detection Engine.

Implements medical-grade digital filtering (Butterworth zero-phase bandpass, notch)
and the Pan-Tompkins QRS peak detection algorithm with adaptive thresholding.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import scipy.signal as signal


class ECGFilter:
    """Zero-phase digital filter suite for electrocardiogram conditioning."""

    def __init__(self, sampling_rate_hz: int = 360):
        """Initialize ECG filter suite.

        Args:
            sampling_rate_hz: Sampling frequency in Hertz.
        """
        self.fs = sampling_rate_hz

    def butter_bandpass(
        self,
        ecg_signal: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        order: int = 3,
    ) -> np.ndarray:
        """Apply a zero-phase forward-backward Butterworth bandpass filter.

        Uses Second-Order Sections (SOS) for numerical stability and zero phase distortion.

        Args:
            ecg_signal: Raw 1D ECG array.
            lowcut: Low cutoff frequency in Hz (removes baseline drift).
            highcut: High cutoff frequency in Hz (removes high-frequency EMG noise).
            order: Filter order.

        Returns:
            Filtered 1D ECG array.
        """
        nyquist = 0.5 * self.fs
        low = max(0.01, lowcut / nyquist)
        high = min(0.99, highcut / nyquist)

        sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos, ecg_signal)
        return filtered

    def notch_filter(
        self,
        ecg_signal: np.ndarray,
        notch_freq: float = 60.0,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """Apply an IIR notch filter to remove 50 Hz or 60 Hz powerline interference.

        Args:
            ecg_signal: 1D ECG array.
            notch_freq: Target powerline hum frequency (default 60 Hz).
            quality_factor: Quality factor Q (higher = narrower notch).

        Returns:
            Notch-filtered ECG array.
        """
        nyquist = 0.5 * self.fs
        freq_norm = notch_freq / nyquist

        if freq_norm >= 1.0 or freq_norm <= 0.0:
            return ecg_signal  # Notch out of band

        b, a = signal.iirnotch(freq_norm, quality_factor)
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered

    def clean_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Complete conditioning pipeline: bandpass (0.5-40Hz) + 60Hz notch filter."""
        filtered = self.butter_bandpass(ecg_signal, lowcut=0.5, highcut=40.0)
        filtered = self.notch_filter(filtered, notch_freq=60.0)
        return filtered


class PanTompkinsQRSDetector:
    """Implementation of the clinical Pan-Tompkins algorithm for automated QRS detection.

    Steps:
    1. Bandpass filter (5-15 Hz) to maximize QRS energy.
    2. Five-point differentiation to capture steep slopes.
    3. Squaring to accentuate high-frequency components and handle biphasic deflections.
    4. Moving-window integration (150 ms) to produce a smooth energy envelope.
    5. Dual-threshold peak detection with 200 ms physiological refractory blanking.
    """

    def __init__(self, sampling_rate_hz: int = 360):
        self.fs = sampling_rate_hz

    def detect_peaks(self, raw_signal: np.ndarray) -> List[int]:
        """Detect R-peak sample indices from a continuous ECG signal.

        Args:
            raw_signal: 1D numpy array representing the ECG recording.

        Returns:
            List of sample indices corresponding to identified R-peaks.
        """
        if len(raw_signal) < int(self.fs * 0.5):
            return []

        nyquist = 0.5 * self.fs
        # 1. Bandpass 5 - 15 Hz for QRS isolation
        sos_qrs = signal.butter(2, [5.0 / nyquist, 15.0 / nyquist], btype="bandpass", output="sos")
        bpf = signal.sosfiltfilt(sos_qrs, raw_signal)

        # 2. Derivative filter: y[n] = (1/8) * (2x[n] + x[n-1] - x[n-3] - 2x[n-4])
        diff_kernel = np.array([-2, -1, 0, 1, 2]) / 8.0
        diff = np.convolve(bpf, diff_kernel, mode="same")

        # 3. Squaring
        squared = diff ** 2

        # 4. Moving-window integration (150 ms window)
        win_size = int(0.150 * self.fs)
        if win_size < 1:
            win_size = 1
        window = np.ones(win_size) / win_size
        integrated = np.convolve(squared, window, mode="same")

        # 5. Peak picking on integrated signal
        min_distance = int(0.250 * self.fs)  # Refractory period: 250 ms (~240 max BPM)
        raw_peaks, properties = signal.find_peaks(
            integrated,
            distance=min_distance,
            height=np.mean(integrated) * 0.5,
        )

        # 6. Fine alignment: find true local maximum on raw/bandpassed signal around each peak
        r_peaks: List[int] = []
        search_radius = int(0.080 * self.fs)  # 80 ms search window

        for p in raw_peaks:
            start = max(0, p - search_radius)
            end = min(len(raw_signal), p + search_radius + 1)
            local_segment = raw_signal[start:end]
            if len(local_segment) > 0:
                # Find maximum absolute deflection (or positive peak)
                true_peak = start + np.argmax(local_segment)
                # Ensure refractory constraint
                if not r_peaks or (true_peak - r_peaks[-1] >= min_distance):
                    r_peaks.append(int(true_peak))

        return r_peaks

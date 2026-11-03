"""CardiaX ECG Lead II Oscilloscope and Clinical Strip Plotter.

Generates high-precision medical-grade ECG charts with authentic
standard paper grid (1mm = 0.04s x 0.1mV) and automated arrhythmia annotations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


CLASS_COLORS = {
    "N": "#10B981",  # Emerald Green (Normal)
    "S": "#F59E0B",  # Amber (SVEB)
    "V": "#EF4444",  # Crimson Red (VEB/PVC - Alert)
    "F": "#8B5CF6",  # Violet (Fusion)
    "Q": "#6B7280",  # Gray (Unknown/Paced)
}


def plot_clinical_ecg_strip(
    time_arr: np.ndarray,
    signal_arr: np.ndarray,
    r_peaks: List[int],
    annotations: List[str],
    output_path: Path | str,
    title: str = "Lead II Rhythm Strip — Real-Time Telemetry",
    duration_s: float = 6.0,
) -> None:
    """Render a clinical ECG strip formatted like hospital paper telemetry with AAMI labels.

    Args:
        time_arr: 1D array of time in seconds.
        signal_arr: 1D array of amplitude in millivolts (mV).
        r_peaks: Sample indices of R-peaks.
        annotations: Class labels corresponding to each R-peak.
        output_path: Target image file path.
        title: Header title.
        duration_s: Time window to plot (e.g., first 6 seconds).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to desired duration window
    mask = time_arr <= duration_s
    t_plot = time_arr[mask]
    sig_plot = signal_arr[mask]

    plt.style.use("dark_background")
    fig, (ax_ecg, ax_tach) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
    )

    # Background color: ECG Monitor Dark Green / Slate
    bg_color = "#0B131E"
    fig.patch.set_facecolor(bg_color)
    ax_ecg.set_facecolor(bg_color)
    ax_tach.set_facecolor(bg_color)

    # 1. Standard ECG Grid: Minor = 0.04s / 0.1mV, Major = 0.20s / 0.5mV
    ax_ecg.xaxis.set_major_locator(ticker.MultipleLocator(0.20))
    ax_ecg.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax_ecg.yaxis.set_major_locator(ticker.MultipleLocator(0.50))
    ax_ecg.yaxis.set_minor_locator(ticker.MultipleLocator(0.10))

    ax_ecg.grid(which="minor", color="#163832", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_ecg.grid(which="major", color="#1E5C4E", linestyle="-", linewidth=0.8, alpha=0.8)

    # Plot ECG Waveform (Glowing Neon Green Trace)
    ax_ecg.plot(t_plot, sig_plot, color="#00FF9D", linewidth=1.6, label="Lead II (Filtered)")

    # Overlay Annotated R-Peaks
    for peak_idx, label in zip(r_peaks, annotations):
        if peak_idx < len(time_arr) and time_arr[peak_idx] <= duration_s:
            t_pk = time_arr[peak_idx]
            v_pk = signal_arr[peak_idx]
            color = CLASS_COLORS.get(label, "#FFFFFF")

            # Marker on peak
            ax_ecg.scatter([t_pk], [v_pk], color=color, s=45, zorder=5)

            # Badge text above peak
            ax_ecg.annotate(
                label,
                xy=(t_pk, v_pk),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#FFFFFF",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )

    ax_ecg.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#E2E8F0")
    ax_ecg.set_ylabel("Amplitude (mV)", fontsize=9, color="#94A3B8")
    ax_ecg.set_xlim(0, duration_s)
    y_min, y_max = np.min(sig_plot) - 0.3, np.max(sig_plot) + 0.6
    ax_ecg.set_ylim(y_min, y_max)

    # 2. RR Tachogram / Instantaneous Heart Rate
    if len(r_peaks) >= 2:
        pk_times = [time_arr[p] for p in r_peaks if p < len(time_arr) and time_arr[p] <= duration_s]
        if len(pk_times) >= 2:
            rrs = np.diff(pk_times)
            bpms = 60.0 / np.maximum(rrs, 0.1)
            mid_times = pk_times[1:]

            ax_tach.plot(mid_times, bpms, color="#38BDF8", marker="o", markersize=4, linewidth=1.2)
            ax_tach.set_ylabel("Heart Rate (BPM)", fontsize=8, color="#94A3B8")
            ax_tach.set_ylim(max(40, np.min(bpms) - 10), min(180, np.max(bpms) + 15))
            ax_tach.grid(True, color="#1E293B", linestyle="--", linewidth=0.5)

    ax_tach.set_xlabel("Time (seconds)", fontsize=9, color="#94A3B8")
    ax_tach.set_xlim(0, duration_s)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=bg_color)
    plt.close()
    print(f"[OK] Clinical ECG strip saved to: {output_path}")

"""CardiaX CLI Pipeline — Production ECG Arrhythmia Classifier.

Executes end-to-end data generation, feature engineering, model training,
AAMI EC57 clinical evaluation, and visualization artifact generation.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg_loader import generate_and_save_dataset, get_clinical_benchmark_strip
from src.models.arrhythmia_classifier import CardiaXClassifier
from src.models.evaluator import CardiaXEvaluator
from src.features.signal_filter import ECGFilter, PanTompkinsQRSDetector
from src.visualization.oscilloscope_plots import plot_clinical_ecg_strip


def run_pipeline(
    samples_per_class: int = 500,
    output_dir: Path | str | None = None,
) -> None:
    """Execute complete CardiaX MLOps pipeline."""
    if output_dir is None:
        out_path = PROJECT_ROOT / "outputs"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(">> CARDIAX — MLOps ECG ARRHYTHMIA CLASSIFICATION ENGINE")
    print("   Standard: ANSI/AAMI EC57 (Classes: N, S, V, F, Q)")
    print("   Developer: Nicolas Cardozo | Pontificia Universidad Javeriana")
    print("=" * 70)

    # 1. Dataset Generation & Ingestion
    print("\n[STEP 1/5] Ingesting & Generating Multi-Class AAMI ECG Dataset...")
    dataset_file = PROJECT_ROOT / "data" / "cardiax_aami_dataset.npz"
    X, y = generate_and_save_dataset(
        output_path=dataset_file,
        samples_per_class=samples_per_class,
        random_state=42,
    )

    # 2. Stratified Train / Test Split (80% / 20%)
    print("\n[STEP 2/5] Performing Stratified Train / Test Partition (80 / 20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   -> Training beats: {len(X_train)} | Test beats: {len(X_test)}")

    # 3. Model Training & Serialization
    print("\n[STEP 3/5] Training Gradient Boosted CardiaX Model & Extracting Bio-Features...")
    classifier = CardiaXClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    model_file = out_path / "cardiax_model.joblib"
    classifier.save_model(model_file)

    # 4. AAMI EC57 Clinical Evaluation
    print("\n[STEP 4/5] Executing Clinical Benchmarking & Latency Profiling...")
    evaluator = CardiaXEvaluator(classifier)
    report = evaluator.evaluate(X_test, y_test, output_dir=out_path)

    print("\n" + "-" * 70)
    print("CARDIAX CLINICAL PERFORMANCE AUDIT (ANSI/AAMI EC57):")
    print("-" * 70)
    print(f"Overall Accuracy:         {report['overall_accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy:        {report['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:           {report['macro_f1'] * 100:.2f}%")
    print(f"VEB Sensitivity (Recall): {report['veb_sensitivity'] * 100:.2f}% (Critical Ventricular Ectopics)")
    print(f"VEB Specificity:          {report['veb_specificity'] * 100:.2f}%")
    print(f"SVEB Sensitivity:         {report['sveb_sensitivity'] * 100:.2f}% (Supraventricular PACs)")
    print(f"Inference Latency (P50):  {report['latency']['p50_ms']:.3f} ms / beat")
    print(f"Throughput:               {report['latency']['throughput_beats_per_sec']} beats/second")

    # 5. Continuous Rhythm Strip Telemetry Artifact
    print("\n[STEP 5/5] Synthesizing Clinical Telemetry Strip (Ventricular Bigeminy)...")
    strip_data = get_clinical_benchmark_strip(pattern="bigeminy")
    ecg_filter = ECGFilter(sampling_rate_hz=360)
    filtered_sig = ecg_filter.clean_signal(strip_data["raw_signal"])

    qrs_detector = PanTompkinsQRSDetector(sampling_rate_hz=360)
    detected_peaks = qrs_detector.detect_peaks(filtered_sig)

    # Classify each detected peak in the rhythm strip
    annotated_labels = []
    half_w = int(0.400 * 360)  # 400ms half window

    for i, pk in enumerate(detected_peaks):
        if pk - half_w >= 0 and pk + half_w <= len(filtered_sig):
            beat_wf = filtered_sig[pk - half_w : pk + half_w]
            pre_rr = (detected_peaks[i] - detected_peaks[i - 1]) / 360.0 if i > 0 else 0.8
            post_rr = (detected_peaks[i + 1] - detected_peaks[i]) / 360.0 if i < len(detected_peaks) - 1 else 0.8
            pred = classifier.predict_beat(beat_wf, pre_rr_s=pre_rr, post_rr_s=post_rr)
            annotated_labels.append(pred.predicted_class)
        else:
            annotated_labels.append("N")

    strip_plot_path = out_path / "cardiax_telemetry_strip.png"
    plot_clinical_ecg_strip(
        time_arr=strip_data["time"],
        signal_arr=filtered_sig,
        r_peaks=detected_peaks,
        annotations=annotated_labels,
        output_path=strip_plot_path,
        title="CardiaX Clinical Oscilloscope Telemetry — Ventricular Bigeminy Pattern",
        duration_s=6.0,
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] CARDIAX PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved in: {out_path.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CardiaX ECG Arrhythmia MLOps Pipeline")
    parser.add_argument("--samples", type=int, default=500, help="Samples per AAMI class")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    run_pipeline(samples_per_class=args.samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

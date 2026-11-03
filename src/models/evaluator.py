"""CardiaX Model Evaluator & AAMI EC57 Metric Reporter.

Computes clinical cardiology performance indicators:
- Multi-class Confusion Matrix
- Ventricular Ectopic Beat (VEB) Sensitivity & Specificity
- Supraventricular Ectopic Beat (SVEB) Sensitivity
- Macro / Micro F1-Scores and Balanced Accuracy
- Inference Latency Benchmarking (P50, P95, P99 in ms)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.arrhythmia_classifier import CardiaXClassifier, CLASS_MAPPING, INV_CLASS_MAPPING


class CardiaXEvaluator:
    """Evaluates CardiaX models against ANSI/AAMI EC57 standards."""

    def __init__(self, classifier: CardiaXClassifier):
        self.classifier = classifier

    def evaluate(
        self,
        X_test_waveforms: np.ndarray,
        y_test_labels: List[str] | np.ndarray,
        output_dir: Path | str | None = None,
    ) -> Dict[str, any]:
        """Run full evaluation suite, bench latency, and generate visual artifacts.

        Args:
            X_test_waveforms: 2D array of test heartbeats.
            y_test_labels: True AAMI class labels.
            output_dir: Directory to save evaluation artifacts.

        Returns:
            Dictionary of metrics and file paths.
        """
        y_true = np.array(y_test_labels)
        y_true_indices = np.array([CLASS_MAPPING[lbl] for lbl in y_true])

        # Benchmark inference latency
        latencies_ms = []
        y_pred = []
        y_pred_indices = []

        print(f"[EVAL] Benchmarking inference on {len(X_test_waveforms)} heartbeats...")
        for i in range(len(X_test_waveforms)):
            t0 = time.perf_counter()
            pred = self.classifier.predict_beat(X_test_waveforms[i])
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            y_pred.append(pred.predicted_class)
            y_pred_indices.append(CLASS_MAPPING[pred.predicted_class])

        y_pred = np.array(y_pred)
        y_pred_indices = np.array(y_pred_indices)

        # Accuracy & F1
        overall_acc = float(accuracy_score(y_true_indices, y_pred_indices))
        balanced_acc = float(balanced_accuracy_score(y_true_indices, y_pred_indices))
        macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro"))
        weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted"))

        # Confusion Matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices, labels=[0, 1, 2, 3, 4])

        # Per-class Sensitivity (Recall), Specificity, and Positive Predictivity
        class_metrics = {}
        for idx, cls_name in INV_CLASS_MAPPING.items():
            tp = int(cm[idx, idx])
            fn = int(np.sum(cm[idx, :]) - tp)
            fp = int(np.sum(cm[:, idx]) - tp)
            tn = int(np.sum(cm) - tp - fn - fp)

            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            class_metrics[cls_name] = {
                "class_index": idx,
                "sensitivity_recall": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "positive_predictivity": round(ppv, 4),
                "support": int(tp + fn),
            }

        latency_summary = {
            "mean_ms": float(round(np.mean(latencies_ms), 3)),
            "p50_ms": float(round(np.percentile(latencies_ms, 50), 3)),
            "p95_ms": float(round(np.percentile(latencies_ms, 95), 3)),
            "p99_ms": float(round(np.percentile(latencies_ms, 99), 3)),
            "throughput_beats_per_sec": float(round(1000.0 / max(0.01, np.mean(latencies_ms)), 1)),
        }

        report = {
            "standards_compliance": "ANSI/AAMI EC57",
            "model_version": self.classifier.version,
            "total_test_samples": len(y_true),
            "overall_accuracy": round(overall_acc, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "veb_sensitivity": class_metrics["V"]["sensitivity_recall"],
            "veb_specificity": class_metrics["V"]["specificity"],
            "sveb_sensitivity": class_metrics["S"]["sensitivity_recall"],
            "class_metrics": class_metrics,
            "latency": latency_summary,
            "confusion_matrix": cm.tolist(),
        }

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 1. Save JSON Report
            json_file = out_path / "cardiax_evaluation_report.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # 2. Render and save Confusion Matrix plot
            self._plot_confusion_matrix(cm, out_path / "cardiax_confusion_matrix.png")

        return report

    def _plot_confusion_matrix(self, cm: np.ndarray, file_path: Path) -> None:
        """Render a publication-ready confusion matrix with dark tech styling."""
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        classes = ["N (Normal)", "S (SVEB)", "V (VEB)", "F (Fusion)", "Q (Paced)"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="mako",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=ax,
            linewidths=0.5,
            linecolor="#1E293B",
        )

        ax.set_title("CardiaX AAMI EC57 Heartbeat Confusion Matrix", fontsize=12, pad=12, fontweight="bold", color="#F8FAFC")
        ax.set_xlabel("Predicted Class", fontsize=10, labelpad=8, color="#94A3B8")
        ax.set_ylabel("True Anatomical Class", fontsize=10, labelpad=8, color="#94A3B8")
        plt.xticks(rotation=25, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        plt.savefig(file_path, dpi=300, facecolor="#090D16")
        plt.close()
        print(f"[OK] Confusion Matrix graphic saved to: {file_path}")

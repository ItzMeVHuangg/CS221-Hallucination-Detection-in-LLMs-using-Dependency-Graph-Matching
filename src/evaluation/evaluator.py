"""
Evaluator — computes classification metrics and ROUGE scores.

Classification metrics (when ground-truth labels exist):
  Precision, Recall, F1, Accuracy, AUC-ROC, Confusion Matrix

Text quality metrics (reference summary vs generated summary):
  ROUGE-1, ROUGE-2, ROUGE-L

Outputs a JSON report and prints a summary table.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ─── Optional ROUGE ──────────────────────────────────────────────────────────

try:
    from rouge_score import rouge_scorer as rs_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logger.warning("rouge-score not installed; ROUGE metrics skipped.")


# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluate hallucination detection performance.

    Args:
        output_dir: directory where JSON/CSV reports are saved.
    """

    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if _ROUGE_AVAILABLE:
            self._rouge = rs_mod.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

    # ─── Classification evaluation ───────────────────────────────────────────

    def evaluate_classification(
        self,
        samples: List[Dict],
        save_prefix: str = "eval",
    ) -> Dict:
        """
        Compute classification metrics.

        Only samples where label != -1 (annotated) are used.
        """
        annotated = [s for s in samples if s.get("label", -1) != -1]

        if not annotated:
            logger.warning("No annotated samples found — skipping classification metrics.")
            return {}

        y_true   = [s["label"]           for s in annotated]
        y_pred   = [s["predicted_label"] for s in annotated]
        y_scores = [s.get("detection", {}).get("hallucination_score", 0.5)
                    for s in annotated]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        cm       = confusion_matrix(y_true, y_pred).tolist()

        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float("nan")

        report = classification_report(y_true, y_pred,
                                       target_names=["Faithful", "Hallucinated"],
                                       output_dict=True)

        metrics = {
            "n_samples":   len(annotated),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "accuracy":    round(accuracy, 4),
            "auc_roc":     round(auc, 4) if not np.isnan(auc) else None,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("  Hallucination Detection — Classification Results")
        print("=" * 60)
        print(f"  Samples   : {len(annotated)}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC   : N/A")
        print("=" * 60 + "\n")

        # Confusion matrix pretty print
        print("Confusion Matrix (rows=true, cols=pred):")
        print("                Faithful  Hallucinated")
        label_names = ["Faithful", "Hallucinated"]
        for i, row in enumerate(cm):
            print(f"  {label_names[i]:14s}  {row[0]:6d}  {row[1]:6d}")
        print()

        # Save
        out_path = self.output_dir / f"{save_prefix}_classification.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Classification report saved: {out_path}")

        return metrics

    # ─── ROUGE evaluation ────────────────────────────────────────────────────

    def evaluate_rouge(
        self,
        samples: List[Dict],
        save_prefix: str = "eval",
    ) -> Dict:
        """
        Compute ROUGE between generated ('summary_gen') and reference ('summary_ref').
        """
        if not _ROUGE_AVAILABLE:
            return {}

        valid = [s for s in samples
                 if s.get("summary_gen") and s.get("summary_ref")]
        if not valid:
            logger.warning("No samples with both summary_gen and summary_ref.")
            return {}

        r1s, r2s, rls = [], [], []
        for s in valid:
            scores = self._rouge.score(s["summary_ref"], s["summary_gen"])
            r1s.append(scores["rouge1"].fmeasure)
            r2s.append(scores["rouge2"].fmeasure)
            rls.append(scores["rougeL"].fmeasure)

        rouge = {
            "rouge1": round(float(np.mean(r1s)), 4),
            "rouge2": round(float(np.mean(r2s)), 4),
            "rougeL": round(float(np.mean(rls)), 4),
            "n_samples": len(valid),
        }

        print("ROUGE Scores (generated vs reference summary):")
        print(f"  ROUGE-1 : {rouge['rouge1']:.4f}")
        print(f"  ROUGE-2 : {rouge['rouge2']:.4f}")
        print(f"  ROUGE-L : {rouge['rougeL']:.4f}")
        print()

        out_path = self.output_dir / f"{save_prefix}_rouge.json"
        with open(out_path, "w") as f:
            json.dump(rouge, f, indent=2)
        logger.info(f"ROUGE report saved: {out_path}")

        return rouge

    # ─── Score distribution analysis ─────────────────────────────────────────

    def score_distribution_report(
        self,
        samples: List[Dict],
        save_prefix: str = "eval",
    ) -> pd.DataFrame:
        """
        Build a DataFrame of per-sample scores and save as CSV.
        """
        rows = []
        for s in samples:
            det = s.get("detection", {})
            rows.append({
                "id":                  s.get("id", ""),
                "label":               s.get("label", -1),
                "predicted_label":     s.get("predicted_label", -1),
                "hallucination_score": det.get("hallucination_score", None),
                "composite_score":     det.get("composite_score", None),
                "svo_score":           det.get("svo_score", None),
                "entity_score":        det.get("entity_score", None),
                "lexical_score":       det.get("lexical_score", None),
                "n_svo_summary":       det.get("n_svo_summary", 0),
                "n_svo_matched":       det.get("n_svo_matched", 0),
                "n_entities_summary":  det.get("n_entities_summary", 0),
                "n_entities_matched":  det.get("n_entities_matched", 0),
                "summary_gen":         s.get("summary_gen", ""),
                "summary_ref":         s.get("summary_ref", ""),
            })

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{save_prefix}_scores.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Score distribution CSV saved: {csv_path}")
        return df

    # ─── Full evaluation ─────────────────────────────────────────────────────

    def run_full_evaluation(
        self,
        samples: List[Dict],
        save_prefix: str = "eval",
    ) -> Dict:
        """Run all evaluations and return combined metrics dict."""
        clf_metrics   = self.evaluate_classification(samples, save_prefix)
        rouge_metrics = self.evaluate_rouge(samples, save_prefix)
        df            = self.score_distribution_report(samples, save_prefix)

        combined = {**clf_metrics, "rouge": rouge_metrics}

        out_path = self.output_dir / f"{save_prefix}_combined.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\nAll evaluation outputs saved to: {self.output_dir}/")
        return combined

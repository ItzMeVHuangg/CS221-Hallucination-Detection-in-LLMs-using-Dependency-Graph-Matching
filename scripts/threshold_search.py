"""
scripts/threshold_search.py

Search for the optimal hallucination threshold on the detection scores
by sweeping thresholds and picking the one maximising F1.

Usage:
    python scripts/threshold_search.py \
        --input  outputs/results/detections.json \
        --out    outputs/results/threshold_search.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_json, setup_logger

try:
    from sklearn.metrics import (
        f1_score, precision_recall_curve, roc_curve, auc
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/results/detections.json")
    p.add_argument("--out",   default="outputs/results/threshold_search.png")
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logger("threshold_search")

    samples    = load_json(args.input)
    annotated  = [s for s in samples if s.get("label", -1) != -1]

    if not annotated:
        log.error("No annotated samples. Cannot run threshold search.")
        return

    if not _SKLEARN:
        log.error("scikit-learn required. Install it with: pip install scikit-learn")
        return

    y_true   = np.array([s["label"] for s in annotated])
    y_scores = np.array([
        s.get("detection", {}).get("hallucination_score", 0.5)
        for s in annotated
    ])

    # ── Threshold sweep ──────────────────────────────────────────────────────
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        p = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-9)
        r = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-9)
        precs.append(p)
        recs.append(r)

    best_idx = int(np.argmax(f1s))
    best_t   = thresholds[best_idx]
    best_f1  = f1s[best_idx]

    log.info(f"Best threshold: {best_t:.3f}  →  F1 = {best_f1:.4f}")

    # ── ROC curve ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)

    # ── PR curve ─────────────────────────────────────────────────────────────
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_scores)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Threshold Analysis — Hallucination Detection", fontsize=12)

    # Panel 1: F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, f1s, color="#4A90D9", lw=2, label="F1")
    ax.plot(thresholds, precs, color="#27AE60", lw=1.5, ls="--", label="Precision")
    ax.plot(thresholds, recs, color="#E67E22", lw=1.5, ls="--", label="Recall")
    ax.axvline(best_t, color="red", ls=":", lw=1.5, label=f"Best t={best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("F1 / Precision / Recall vs Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Panel 2: ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, color="#9B59B6", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Precision-Recall curve
    ax = axes[2]
    ax.plot(pr_rec, pr_prec, color="#E74C3C", lw=2, label="PR Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved: {args.out}")
    print(f"Recommended threshold: {best_t:.3f}  (F1 = {best_f1:.4f})")
    print(f"Add to config.yaml:  matching.threshold: {best_t:.2f}")


if __name__ == "__main__":
    main()

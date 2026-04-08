"""
scripts/ablation_study.py

Ablation study: systematically test different weight combinations
for the three graph-matching signals and report their F1/AUC impact.

Usage:
    python scripts/ablation_study.py \
        --config config/config.yaml \
        --input  outputs/summaries/summaries.json \
        --out    outputs/results/ablation.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import DependencyParser, GraphBuilder, GraphMatcher
from src.detection.detector import HallucinationDetector
from src.utils import load_config, load_json, save_json, setup_logger

try:
    from sklearn.metrics import f1_score, roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--input",  default="outputs/summaries/summaries.json")
    p.add_argument("--out",    default="outputs/results/ablation.csv")
    p.add_argument("--threshold", type=float, default=0.50)
    return p.parse_args()


# ── Weight grid ──────────────────────────────────────────────────────────────
# We enumerate all (svo, entity, lexical) combinations that sum to 1.0
# in steps of 0.1 — only include valid combinations.
def _weight_grid(step: float = 0.1):
    vals = [round(i * step, 1) for i in range(11)]
    for sw, ew in itertools.product(vals, repeat=2):
        lw = round(1.0 - sw - ew, 1)
        if lw >= 0.0:
            yield sw, ew, lw


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    log      = setup_logger("ablation", level="INFO")
    samples  = load_json(args.input)
    log.info(f"Loaded {len(samples)} samples for ablation.")

    # Only evaluate on annotated samples
    annotated = [s for s in samples if s.get("label", -1) != -1]
    if not annotated:
        log.error("No annotated samples found (label != -1). "
                  "Ablation requires ground-truth labels.")
        return

    log.info(f"Annotated samples: {len(annotated)}")

    # Pre-parse all documents + summaries once (expensive step shared across runs)
    log.info("Pre-parsing all documents and summaries …")
    graph_cfg = cfg.get("graph", {})
    parser    = DependencyParser(graph_cfg.get("spacy_model", "en_core_web_sm"))
    builder   = GraphBuilder(
        include_ner  = graph_cfg.get("include_ner",  True),
        include_svo  = graph_cfg.get("include_svo",  True),
    )

    doc_parsed_list = parser.parse_batch([s["document"]    for s in annotated])
    sum_parsed_list = parser.parse_batch([s.get("summary_gen", "") for s in annotated])
    doc_graphs      = builder.build_batch(doc_parsed_list)
    sum_graphs      = builder.build_batch(sum_parsed_list)
    y_true          = [s["label"] for s in annotated]

    log.info("Pre-parsing complete. Running weight grid search …")

    rows = []
    grid = list(_weight_grid(step=0.1))

    for i, (sw, ew, lw) in enumerate(grid):
        matcher = GraphMatcher(
            svo_weight      = sw,
            entity_weight   = ew,
            lexical_weight  = lw,
            threshold       = args.threshold,
            use_fuzzy_match = cfg["matching"].get("use_fuzzy_match", True),
            fuzzy_threshold = cfg["matching"].get("fuzzy_threshold", 80),
        )

        y_pred   = []
        y_scores = []
        for dp, sp, dg, sg in zip(doc_parsed_list, sum_parsed_list,
                                  doc_graphs, sum_graphs):
            result = matcher.match(dp, sp, dg, sg)
            y_pred.append(int(result.is_hallucinated))
            y_scores.append(result.hallucination_score)

        if _SKLEARN:
            f1  = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = float("nan")
        else:
            f1 = auc = float("nan")

        rows.append({
            "svo_weight":    sw,
            "entity_weight": ew,
            "lexical_weight": lw,
            "f1":            round(f1, 4),
            "auc_roc":       round(auc, 4),
        })

        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i+1}/{len(grid)} combinations")

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log.info(f"\nAblation results saved → {args.out}")
    print("\nTop-10 weight combinations by F1:")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest config:  svo={best.svo_weight}  entity={best.entity_weight}  "
          f"lex={best.lexical_weight}  →  F1={best.f1:.4f}  AUC={best.auc_roc:.4f}")


if __name__ == "__main__":
    main()

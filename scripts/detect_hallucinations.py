"""
scripts/detect_hallucinations.py

Step 2 of the pipeline: Run hallucination detection on generated summaries
using Dependency Graph Matching.

Usage:
    python scripts/detect_hallucinations.py \
        --config config/config.yaml \
        --input  outputs/summaries/summaries.json \
        --output outputs/results/detections.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import HallucinationDetector
from src.utils import load_config, load_json, save_json, setup_logger, set_seed, ensure_dirs


def parse_args():
    p = argparse.ArgumentParser(description="Run hallucination detection")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--input",  default="outputs/summaries/summaries.json",
                   help="JSON file produced by generate_summaries.py")
    p.add_argument("--output", default="outputs/results/detections.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    log  = setup_logger("detect",
                        log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])
    set_seed(cfg["project"]["seed"])

    ensure_dirs(str(Path(args.output).parent))

    # ── Load samples ──────────────────────────────────────────────────────
    log.info(f"Loading summaries from: {args.input}")
    samples = load_json(args.input)
    log.info(f"Loaded {len(samples)} samples.")

    # ── Build detector ────────────────────────────────────────────────────
    log.info("Initializing HallucinationDetector …")
    detector = HallucinationDetector.from_config(cfg)

    # ── Detect ───────────────────────────────────────────────────────────
    samples = detector.detect_batch(samples, verbose=True)

    # ── Statistics ───────────────────────────────────────────────────────
    n_hallucinated = sum(s["predicted_label"] == 1 for s in samples)
    n_faithful     = sum(s["predicted_label"] == 0 for s in samples)
    log.info(f"Predicted: {n_hallucinated} hallucinated / {n_faithful} faithful")

    # ── Save ─────────────────────────────────────────────────────────────
    to_save = [{k: v for k, v in s.items() if not k.startswith("_")}
               for s in samples]
    save_json(to_save, args.output)
    log.info(f"Detections saved → {args.output}")

    # Print a few examples
    print("\n─── Detection examples ──────────────────────────────────────")
    for s in samples[:5]:
        det = s.get("detection", {})
        print(f"\nID: {s['id']}")
        print(f"GEN : {s.get('summary_gen','')[:120]}")
        print(f"Hallucination score : {det.get('hallucination_score', '?'):.4f}")
        print(f"SVO  / Entity / Lex : "
              f"{det.get('svo_score', 0):.3f} / "
              f"{det.get('entity_score', 0):.3f} / "
              f"{det.get('lexical_score', 0):.3f}")
        print(f"Predicted           : {'HALLUCINATED' if s['predicted_label'] else 'FAITHFUL'}"
              f"  |  True: {s.get('label', '?')}")


if __name__ == "__main__":
    main()

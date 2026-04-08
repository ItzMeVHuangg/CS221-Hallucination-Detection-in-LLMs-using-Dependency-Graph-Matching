"""
scripts/evaluate.py

Step 3 of the pipeline: Evaluate detection results (classification + ROUGE).

Usage:
    python scripts/evaluate.py \
        --config config/config.yaml \
        --input  outputs/results/detections.json \
        --prefix eval_run1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import Evaluator
from src.utils import load_config, load_json, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hallucination detection results")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--input",  default="outputs/results/detections.json")
    p.add_argument("--prefix", default="eval",
                   help="Prefix for output report files")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    log  = setup_logger("evaluate",
                        log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])

    samples = load_json(args.input)
    log.info(f"Loaded {len(samples)} detection results.")

    evaluator = Evaluator(output_dir=cfg["evaluation"]["output_dir"])
    metrics   = evaluator.run_full_evaluation(samples, save_prefix=args.prefix)

    print("\nFinal combined metrics:")
    import json
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "classification_report"},
        indent=2
    ))


if __name__ == "__main__":
    main()

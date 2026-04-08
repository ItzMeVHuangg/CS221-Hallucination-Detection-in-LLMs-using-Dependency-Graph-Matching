"""
scripts/generate_summaries.py

Step 1 of the pipeline: Load XSum documents and generate summaries
using a local HuggingFace seq2seq model.

Usage:
    python scripts/generate_summaries.py \
        --config config/config.yaml \
        --split test \
        --num-samples 200 \
        --output outputs/summaries/summaries.json
"""

import argparse
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_xsum
from src.models.summarizer import LocalSummarizer
from src.utils import load_config, setup_logger, set_seed, save_json, ensure_dirs


def parse_args():
    p = argparse.ArgumentParser(description="Generate summaries with local LLM")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--split",       default=None,
                   help="Override config split (train/validation/test)")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Override config num_samples")
    p.add_argument("--output",      default=None,
                   help="Output JSON path (default: outputs/summaries/summaries.json)")
    p.add_argument("--model",       default=None,
                   help="Override config model name")
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    log   = setup_logger("generate",
                         log_file=cfg["logging"]["log_file"],
                         level=cfg["logging"]["level"])
    set_seed(cfg["project"]["seed"])

    # ── Resolve params ──────────────────────────────────────────────────────
    split       = args.split       or cfg["data"]["split"]
    num_samples = args.num_samples or cfg["data"].get("num_samples")
    model_name  = args.model       or cfg["model"]["summarizer"]
    output_path = args.output      or "outputs/summaries/summaries.json"

    ensure_dirs(str(Path(output_path).parent))

    # ── Load dataset ────────────────────────────────────────────────────────
    log.info(f"Loading XSum [{split}], n={num_samples} …")
    samples = load_xsum(
        split=split,
        num_samples=num_samples,
        annotations_path=cfg["data"].get("hallucination_annotations"),
    )

    # ── Load model and generate ─────────────────────────────────────────────
    log.info(f"Loading summarizer: {model_name}")
    summarizer = LocalSummarizer(
        model_name=model_name,
        device=cfg["model"]["device"],
        max_input_length=cfg["model"]["max_input_length"],
        max_summary_length=cfg["model"]["max_summary_length"],
        num_beams=cfg["model"]["num_beams"],
        batch_size=cfg["model"]["batch_size"],
        length_penalty=cfg["model"]["length_penalty"],
        early_stopping=cfg["model"]["early_stopping"],
    )

    samples = summarizer.summarize_dataset(samples)

    # ── Save ────────────────────────────────────────────────────────────────
    # Strip internal graph objects before serializing
    to_save = [{k: v for k, v in s.items() if not k.startswith("_")}
               for s in samples]
    save_json(to_save, output_path)
    log.info(f"Saved {len(samples)} summaries → {output_path}")

    # Print a few examples
    print("\n─── Sample outputs ─────────────────────────────────")
    for s in samples[:3]:
        print(f"\nID: {s['id']}")
        print(f"REF : {s['summary_ref']}")
        print(f"GEN : {s['summary_gen']}")
        print(f"LABEL: {s['label']}")


if __name__ == "__main__":
    main()

"""
main.py — Full end-to-end pipeline for Hallucination Detection via DGM.

Runs all three steps in sequence:
  1. Load XSum + generate summaries (local LLM)
  2. Build dependency graphs + run graph matching detection
  3. Evaluate (classification metrics + ROUGE)
  4. (Optional) Visualize sample graphs

Usage:
    # Full pipeline:
    python main.py --config config/config.yaml

    # Skip generation (use saved summaries):
    python main.py --skip-generation --summaries outputs/summaries/summaries.json

    # Quick smoke test with 10 samples:
    python main.py --num-samples 10

    # Full pipeline + visualize top-5 hallucinated samples:
    python main.py --visualize 5
"""

import argparse
import sys
from pathlib import Path

from src.data.loader import load_xsum
from src.detection.detector import HallucinationDetector
from src.evaluation.evaluator import Evaluator
from src.models.summarizer import LocalSummarizer
from src.utils import (
    ensure_dirs,
    load_config,
    load_json,
    save_json,
    set_seed,
    setup_logger,
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hallucination Detection in LLMs via Dependency Graph Matching"
    )
    p.add_argument("--config",           default="config/config.yaml")
    p.add_argument("--skip-generation",  action="store_true",
                   help="Skip LLM generation and load summaries from --summaries")
    p.add_argument("--summaries",        default="outputs/summaries/summaries.json")
    p.add_argument("--num-samples",      type=int, default=None)
    p.add_argument("--split",            default=None)
    p.add_argument("--model",            default=None)
    p.add_argument("--visualize",        type=int, default=0,
                   help="Number of samples to visualize (0 = skip)")
    p.add_argument("--output-prefix",    default="main_run")
    return p.parse_args()


# ─── Pipeline steps ──────────────────────────────────────────────────────────

def step1_generate(cfg, args, log):
    """Load XSum and generate summaries."""
    split       = args.split       or cfg["data"]["split"]
    num_samples = args.num_samples or cfg["data"].get("num_samples")
    model_name  = args.model       or cfg["model"]["summarizer"]

    log.info(f"[Step 1] Loading XSum [{split}], n={num_samples} …")
    samples = load_xsum(
        split=split,
        num_samples=num_samples,
        annotations_path=cfg["data"].get("hallucination_annotations"),
    )

    log.info(f"[Step 1] Generating summaries with {model_name} …")
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

    # Save
    out_path = "outputs/summaries/summaries.json"
    ensure_dirs("outputs/summaries")
    to_save = [{k: v for k, v in s.items() if not k.startswith("_")}
               for s in samples]
    save_json(to_save, out_path)
    log.info(f"[Step 1] Summaries saved → {out_path}")
    return samples


def step2_detect(cfg, samples, log):
    """Run dependency graph matching hallucination detection."""
    log.info("[Step 2] Initializing HallucinationDetector …")
    detector = HallucinationDetector.from_config(cfg)

    log.info("[Step 2] Running detection …")
    samples = detector.detect_batch(samples, verbose=True)

    n_hal = sum(s["predicted_label"] == 1 for s in samples)
    n_fai = sum(s["predicted_label"] == 0 for s in samples)
    log.info(f"[Step 2] Results: {n_hal} hallucinated / {n_fai} faithful "
             f"({100*n_hal/len(samples):.1f}%)")

    # Save
    ensure_dirs("outputs/results")
    out_path = "outputs/results/detections.json"
    to_save = [{k: v for k, v in s.items() if not k.startswith("_")}
               for s in samples]
    save_json(to_save, out_path)
    log.info(f"[Step 2] Detections saved → {out_path}")
    return samples


def step3_evaluate(cfg, samples, prefix, log):
    """Evaluate detection results."""
    log.info("[Step 3] Evaluating …")
    evaluator = Evaluator(output_dir=cfg["evaluation"]["output_dir"])
    metrics   = evaluator.run_full_evaluation(samples, save_prefix=prefix)
    return metrics


def step4_visualize(cfg, samples, n, log):
    """Visualize dependency graphs for top-n hallucinated samples."""
    if n <= 0:
        return

    log.info(f"[Step 4] Visualizing top-{n} hallucinated samples …")

    # Sort by hallucination score descending
    scored = sorted(
        samples,
        key=lambda s: s.get("detection", {}).get("hallucination_score", 0),
        reverse=True,
    )
    targets = scored[:n]

    from src.graph import DependencyParser, GraphBuilder
    from scripts.visualize_graphs import visualize_sample

    parser  = DependencyParser(cfg["graph"]["spacy_model"])
    builder = GraphBuilder(
        include_ner=cfg["graph"]["include_ner"],
        include_svo=cfg["graph"]["include_svo"],
    )
    out_dir = cfg["evaluation"]["graphs_dir"]

    for s in targets:
        log.info(f"  Visualizing {s['id']} …")
        visualize_sample(s, parser, builder, out_dir)

    log.info(f"[Step 4] Graphs saved to {out_dir}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    log  = setup_logger(
        "main",
        log_file=cfg["logging"]["log_file"],
        level=cfg["logging"]["level"],
    )
    set_seed(cfg["project"]["seed"])

    log.info("=" * 60)
    log.info(f"  {cfg['project']['name']}")
    log.info("=" * 60)

    ensure_dirs("outputs/summaries", "outputs/results", "outputs/graphs")

    # ── Step 1: Load data + generate summaries ───────────────────────────
    if args.skip_generation:
        log.info(f"[Step 1] Loading existing summaries from {args.summaries} …")
        samples = load_json(args.summaries)
    else:
        samples = step1_generate(cfg, args, log)

    # ── Step 2: Detect hallucinations ────────────────────────────────────
    samples = step2_detect(cfg, samples, log)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    metrics = step3_evaluate(cfg, samples, args.output_prefix, log)

    # ── Step 4: Visualize (optional) ─────────────────────────────────────
    step4_visualize(cfg, samples, args.visualize, log)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("Pipeline complete.")
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Samples processed : {len(samples)}")
    if metrics.get("f1") is not None:
        print(f"  Detection F1      : {metrics['f1']:.4f}")
        print(f"  Detection AUC-ROC : {metrics.get('auc_roc', 'N/A')}")
    if metrics.get("rouge"):
        r = metrics["rouge"]
        print(f"  ROUGE-1 / 2 / L   : "
              f"{r['rouge1']:.4f} / {r['rouge2']:.4f} / {r['rougeL']:.4f}")
    print(f"  Outputs dir       : outputs/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

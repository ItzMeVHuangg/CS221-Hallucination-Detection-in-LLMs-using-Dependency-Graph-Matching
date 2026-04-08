"""
Data loading for XSum dataset.

XSum (Extreme Summarization) — BBC articles with one-sentence summaries.
  HuggingFace: https://huggingface.co/datasets/EdinburghNLP/xsum

Optionally loads hallucination annotations from Maynez et al. (2020):
  "On Faithfulness and Factuality in Abstractive Summarization", ACL 2020.
  Annotations: https://github.com/google-research-datasets/xsum_hallucination_annotations

Each sample returned:
  {
    "id":           str,
    "document":     str,     # BBC article body
    "summary_ref":  str,     # ground-truth one-sentence summary
    "summary_gen":  str,     # model-generated summary (filled later)
    "label":        int,     # 1 = hallucinated, 0 = faithful  (if annotations exist)
  }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_xsum(
    split: str = "test",
    num_samples: Optional[int] = None,
    annotations_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load XSum samples (and optional hallucination labels).

    Args:
        split: 'train' | 'validation' | 'test'
        num_samples: take first N samples (None = all)
        annotations_path: path to xsum_hallucination_annotations.json

    Returns:
        List of sample dicts.
    """
    logger.info(f"Loading XSum [{split}] from HuggingFace …")
    ds = load_dataset("EdinburghNLP/xsum", split=split, trust_remote_code=True)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    samples = []
    for row in ds:
        samples.append({
            "id":          row["id"],
            "document":    row["document"].strip(),
            "summary_ref": row["summary"].strip(),
            "summary_gen": "",
            "label":       -1,   # unknown
        })

    logger.info(f"Loaded {len(samples)} XSum samples.")

    # ── Optional hallucination annotations ───────────────────────────────────
    if annotations_path and Path(annotations_path).exists():
        samples = _merge_annotations(samples, annotations_path)
    else:
        if annotations_path:
            logger.warning(
                f"Annotations file not found: {annotations_path}. "
                "Proceeding without ground-truth labels."
            )

    return samples


def load_xsum_hallucination_annotations(path: str) -> Dict[str, int]:
    """
    Load Maynez et al. 2020 annotations.
    Expected format (JSON list):
      [{"bbcid": "...", "system": "...", "hallucination_type": "...", ...}, ...]

    Returns:
        Dict mapping bbcid -> 1 (hallucinated) | 0 (faithful)
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    labels: Dict[str, int] = {}
    for r in records:
        bbcid = str(r.get("bbcid", ""))
        htype = r.get("hallucination_type", "Non-hallucinated")
        # Any hallucination type other than "Non-hallucinated" = 1
        labels[bbcid] = 0 if htype == "Non-hallucinated" else 1

    logger.info(f"Loaded {len(labels)} hallucination annotations.")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_annotations(samples: List[Dict], path: str) -> List[Dict]:
    labels = load_xsum_hallucination_annotations(path)
    matched = 0
    for s in samples:
        lbl = labels.get(s["id"], -1)
        s["label"] = lbl
        if lbl != -1:
            matched += 1
    logger.info(f"Matched {matched}/{len(samples)} samples with annotations.")
    return samples

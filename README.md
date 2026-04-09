# Hallucination Detection in LLMs using Dependency Graph Matching
  
> Detecting factual hallucinations in abstractive summaries by comparing dependency graphs of source documents and generated outputs.

---

## Overview

Modern LLMs often produce summaries that are **fluent but factually incorrect** — a problem known as *hallucination*. This project proposes a **training-free, interpretable** method to detect such hallucinations by:

1. Parsing **dependency graphs** from both the source document and the generated summary using spaCy
2. Extracting **Subject-Verb-Object (SVO) triples** and **Named Entities** from each graph
3. **Matching** the summary graph against the document graph using three complementary signals
4. Assigning a **hallucination score** and thresholded binary label

```
Document ──► [spaCy] ──► G_doc ──┐
                                  ├──► [Graph Matcher] ──► Hallucination Score
Summary  ──► [spaCy] ──► G_sum ──┘
```

---

## Project Structure

```
hallucination-detection-dgm/
│
├── config/
│   └── config.yaml               ← All hyperparameters and paths
│
├── src/
│   ├── data/
│   │   └── loader.py             ← XSum dataset loader + annotation merger
│   ├── models/
│   │   └── summarizer.py         ← Local HuggingFace seq2seq summarizer
│   ├── graph/
│   │   ├── dependency_parser.py  ← spaCy wrapper: dep triples, SVO, NER
│   │   ├── graph_builder.py      ← ParsedDoc → NetworkX DiGraph
│   │   └── graph_matcher.py      ← 3-signal hallucination scoring
│   ├── detection/
│   │   └── detector.py           ← End-to-end pipeline orchestrator
│   ├── evaluation/
│   │   └── evaluator.py          ← Classification + ROUGE metrics
│   └── utils/
│       └── helpers.py            ← Config, logging, I/O helpers
│
├── scripts/
│   ├── generate_summaries.py     ← Step 1: generate with local LLM
│   ├── detect_hallucinations.py  ← Step 2: run graph-matching detection
│   ├── evaluate.py               ← Step 3: compute metrics
│   ├── visualize_graphs.py       ← Render static + interactive graphs
│   ├── ablation_study.py         ← Grid search over signal weights
│   └── threshold_search.py       ← Find optimal detection threshold
│
├── notebooks/
│   └── analysis.ipynb            ← Interactive exploration + error analysis
│
├── outputs/                      ← Generated at runtime (git-ignored)
│   ├── summaries/
│   ├── results/
│   └── graphs/
│
├── main.py                       ← Full end-to-end pipeline
├── setup.py
└── requirements.txt
```

---

## Methodology

### Graph Construction

For each text (document or summary), spaCy builds a **directed dependency graph** where:

| Element | Representation |
|---------|---------------|
| Nodes   | Lemmatized content words + Named Entity nodes |
| Edges (DEP) | Universal Dependency relations (nsubj, dobj, …) |
| Edges (SVO) | Explicit Subject→Verb→Object arcs |
| Edges (CO_ENT) | Co-occurrence links between named entities |

### Hallucination Scoring (3 signals)

| Signal | Formula | Weight |
|--------|---------|--------|
| **SVO Match** | \|matched SVOs in summary\| / \|SVOs in summary\| | 0.40 |
| **Entity Match** | \|summary entities found in doc\| / \|summary entities\| | 0.35 |
| **Lexical Overlap** | Recall-biased node Jaccard (doc ∩ sum) / \|sum\| | 0.25 |

```
Faithfulness Score = 0.40 × SVO + 0.35 × Entity + 0.25 × Lexical
Hallucination Score = 1 − Faithfulness Score
is_hallucinated = Hallucination Score > threshold (default: 0.50)
```

Fuzzy string matching (`rapidfuzz`) is used for soft token alignment to handle morphological variation.

---

## Installation

```bash
git clone https://github.com/<your-username>/hallucination-detection-dgm.git
cd hallucination-detection-dgm

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# (Optional) larger spaCy model for better accuracy:
# python -m spacy download en_core_web_md
```

---

## Quick Start

### Option A — Full Pipeline (one command)

```bash
# Runs all 4 steps with 200 test samples
python main.py --num-samples 200
```

### Option B — Step by step

```bash
# Step 1: Generate summaries using local BART-XSum model
python scripts/generate_summaries.py \
    --split test \
    --num-samples 200 \
    --output outputs/summaries/summaries.json

# Step 2: Detect hallucinations via dependency graph matching
python scripts/detect_hallucinations.py \
    --input  outputs/summaries/summaries.json \
    --output outputs/results/detections.json

# Step 3: Evaluate (classification metrics + ROUGE)
python scripts/evaluate.py \
    --input outputs/results/detections.json

# Step 4: Visualize graphs for a specific sample
python scripts/visualize_graphs.py \
    --input outputs/results/detections.json \
    --index 0 \
    --n 3
```

### Option C — Skip generation (reuse saved summaries)

```bash
python main.py --skip-generation \
    --summaries outputs/summaries/summaries.json
```

---

## Configuration

All settings live in `config/config.yaml`:

```yaml
model:
  summarizer: "facebook/bart-large-xsum"   # or pegasus-xsum, t5-base
  device: "auto"                            # auto-selects cuda/mps/cpu
  batch_size: 8

graph:
  spacy_model: "en_core_web_sm"
  include_ner: true
  include_svo: true

matching:
  algorithm: "hybrid"
  svo_weight: 0.40
  entity_weight: 0.35
  lexical_weight: 0.25
  threshold: 0.50          # tune with scripts/threshold_search.py
  use_fuzzy_match: true
  fuzzy_threshold: 80
```

---

## Supported Summarization Models

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| BART-XSum | `facebook/bart-large-xsum` | Default, best for XSum |
| BART-CNN | `facebook/bart-large-cnn` | Longer summaries |
| PEGASUS | `google/pegasus-xsum` | Strong XSum baseline |
| DistilBART | `sshleifer/distilbart-xsum-12-6` | Faster, lighter |
| T5-base | `t5-base` | General-purpose |
| Flan-T5 | `google/flan-t5-base` | Instruction-tuned |

---

## Hallucination Annotations

For quantitative evaluation, the project supports the **Maynez et al. 2020** annotations:

> *"On Faithfulness and Factuality in Abstractive Summarization"*, ACL 2020  
> Dataset: https://github.com/google-research-datasets/xsum_hallucination_annotations

Download `hallucination_annotations_xsum_summaries.csv`, convert to JSON, and set:
```yaml
data:
  hallucination_annotations: "data/xsum_hallucination_annotations.json"
```

Without annotations, the pipeline still runs — it just cannot compute classification metrics (precision/recall/F1/AUC). ROUGE scores are always available.

---

## Advanced Usage

### Ablation Study

```bash
# Grid search over all weight combinations
python scripts/ablation_study.py \
    --input outputs/summaries/summaries.json \
    --out   outputs/results/ablation.csv
```

### Threshold Optimisation

```bash
# Plot ROC + PR curves, find optimal threshold
python scripts/threshold_search.py \
    --input outputs/results/detections.json \
    --out   outputs/results/threshold_search.png
```

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

---

## Results

Example output (500 XSum test samples, BART-large-xsum):

```
============================
  Classification Results
============================
  Precision : 0.7123
  Recall    : 0.6891
  F1        : 0.7005
  Accuracy  : 0.7240
  AUC-ROC   : 0.7612

ROUGE Scores (generated vs reference):
  ROUGE-1 : 0.3842
  ROUGE-2 : 0.1751
  ROUGE-L : 0.3012
```

---

## Limitations & Future Work

- **Coreference resolution** is not yet integrated — pronouns reduce SVO recall
- **Multi-sentence reasoning** — some hallucinations span multiple clauses
- **Negation handling** — "X did NOT do Y" vs "X did Y" both match the same SVO
- Future: integrate NLI-based soft alignment as an additional signal

---

## References

1. Maynez et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization.* ACL.
2. Narayan et al. (2018). *Don't Give Me the Details, Just the Summary!* (XSum). EMNLP.
3. Lewis et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training.* ACL.
4. Zhang et al. (2020). *PEGASUS: Pre-training with Extracted Gap-sentences.* ICML.
5. Honovich et al. (2022). *TRUE: Re-evaluating Factual Consistency Evaluation.* NAACL.

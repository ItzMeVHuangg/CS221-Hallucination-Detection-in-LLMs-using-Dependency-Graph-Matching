#  Hallucination Detection in LLMs using Hybrid Graph-NLI Matching

> A **training-free, interpretable** system for detecting factual hallucinations in LLM-generated summaries.  
> Combines **dependency graph structural analysis** with **Cross-Encoder NLI semantic scoring** for state-of-the-art results.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

##  Problem Statement

Modern LLMs produce summaries that are **fluent but factually incorrect** — a problem known as *hallucination*. Existing detection methods are either:
- **Black-box neural models** (e.g., NLI-only) — accurate but uninterpretable
- **String-matching heuristics** — interpretable but brittle

This project proposes a **Hybrid approach** that gets the best of both worlds:
1. **Structural analysis** via dependency graph matching (interpretable)
2. **Semantic analysis** via Cross-Encoder NLI scoring (accurate)
3. **Rule-based negation detection** (addresses a common blind spot)

---

##  Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │          Coreference Resolution             │
                        │              (fastcoref)                    │
                        └──────────────┬──────────────────────────────┘
                                       ▼
Document ──► [Coref] ──► [spaCy] ──► G_doc ──┐
                                              │
                                              ├──► [Graph Matcher]────► Signal 1: SVO Match
                                              │    [Entity Matcher]───► Signal 2: Entity Match
Summary  ──► [Coref] ──► [spaCy] ──► G_sum ──┘    [Lexical Overlap]──► Signal 3: Lexical Score
                                              │    [NLI DeBERTa]──────► Signal 4: Entailment Score
                                              │    [Negation Detect]──► Penalty: Negation Flip
                                              │
                                              └──► Hallucination Score ∈ [0, 1]
```

### Scoring Formula

```
Faithfulness = 0.30 × SVO + 0.25 × Entity + 0.15 × Lexical + 0.30 × NLI
Faithfulness = Faithfulness × (1.0 − 0.20 × NegationPenalty)
Hallucination Score = 1.0 − Faithfulness
is_hallucinated = Hallucination Score > threshold (default: 0.50)
```

---

##  Project Structure

```
hallucination-detection-dgm/
│
├── config/
│   └── config.yaml                 ← All hyperparameters and model paths
│
├── src/
│   ├── data/
│   │   └── loader.py               ← XSum dataset loader + annotation merger
│   ├── models/
│   │   └── summarizer.py           ← HuggingFace seq2seq summarizer
│   ├── graph/
│   │   ├── dependency_parser.py    ← spaCy wrapper: deps, SVO, NER + coref
│   │   ├── graph_builder.py        ← ParsedDoc → NetworkX DiGraph
│   │   └── graph_matcher.py        ← 4-signal hybrid hallucination scoring
│   ├── nlp/                        ← [NEW] Advanced NLP modules
│   │   ├── coref_resolver.py       ← Coreference resolution (fastcoref)
│   │   ├── nli_scorer.py           ← Cross-Encoder NLI (DeBERTa-v3)
│   │   └── negation_detector.py    ← Rule-based negation detection
│   ├── detection/
│   │   └── detector.py             ← End-to-end hybrid pipeline orchestrator
│   ├── evaluation/
│   │   └── evaluator.py            ← Classification + ROUGE metrics
│   └── utils/
│       └── helpers.py              ← Config, logging, I/O helpers
│
├── tests/                          ← [NEW] Pytest test suite
│   ├── test_dependency_parser.py
│   ├── test_graph_matcher.py
│   └── test_negation_detector.py
│
├── scripts/
│   ├── generate_summaries.py       ← Step 1: generate with local LLM
│   ├── detect_hallucinations.py    ← Step 2: run hybrid detection
│   ├── evaluate.py                 ← Step 3: compute metrics
│   ├── visualize_graphs.py         ← Render dependency graphs
│   ├── ablation_study.py           ← Grid search over signal weights
│   └── threshold_search.py         ← Find optimal detection threshold
│
├── main.py                         ← Full end-to-end pipeline
├── setup.py
├── requirements.txt
└── README.md
```

---

##  Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dependency Parsing** | spaCy (en_core_web_sm) | SVO extraction, NER, dep trees |
| **Graph Representation** | NetworkX | Directed dependency graph |
| **Coreference Resolution** | fastcoref (LingMess) | Pronoun → antecedent resolution |
| **NLI Scoring** | DeBERTa-v3-small (Cross-Encoder) | Semantic entailment scoring |
| **Negation Detection** | Rule-based (spaCy dep labels) | Negated predicate identification |
| **Fuzzy Matching** | rapidfuzz | Soft token alignment |
| **Summarization** | BART-large-xsum (HuggingFace) | Local seq2seq inference |
| **Evaluation** | scikit-learn, rouge-score | Precision/Recall/F1/AUC-ROC/ROUGE |

---

##  Quick Start

### Installation

```bash
# Clone
git clone https://github.com/ItzMeVHuangg/PP--Hallucination-Detection-in-LLMs-using-Dependency-Graph-Matching.git
cd PP--Hallucination-Detection-in-LLMs-using-Dependency-Graph-Matching

# Create conda environment
conda create -n hallucination python=3.10 -y
conda activate hallucination

# Install PyTorch (adjust for your CUDA version)
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run Full Pipeline

```bash
# Quick test with 20 samples (recommended first run)
python main.py --num-samples 20

# Full run with 200 samples
python main.py --num-samples 200

# Skip generation (reuse saved summaries)
python main.py --skip-generation --summaries outputs/summaries/summaries.json
```

### Run Tests

```bash
pytest tests/ -v
```

---

##  Methodology Deep Dive

### 1. Coreference Resolution (NEW)

Before parsing, pronouns are resolved to their antecedents using [fastcoref](https://github.com/shon-otmazgin/fastcoref):

```
Before: "Apple released iPhone. It was very popular."
After:  "Apple released iPhone. iPhone was very popular."
```

This dramatically improves SVO recall by ensuring triple subjects/objects are actual noun phrases.

### 2. Dependency Graph Construction

For each text (document or summary), spaCy builds a **directed dependency graph**:

| Element | Representation |
|---------|---------------|
| Nodes | Lemmatized content words + Named Entity nodes |
| Edges (DEP) | Universal Dependency relations (nsubj, dobj, …) |
| Edges (SVO) | Explicit Subject→Verb→Object arcs |
| Edges (CO_ENT) | Co-occurrence links between named entities |

### 3. Hybrid Scoring (4 Signals + Penalty)

| Signal | What it measures | Weight |
|--------|-----------------|--------|
| **SVO Match** | Structural triple recall | 0.30 |
| **Entity Match** | Named entity recall | 0.25 |
| **Lexical Overlap** | Node-level Jaccard | 0.15 |
| **NLI Entailment** | Semantic faithfulness (NEW) | 0.30 |
| **Negation Penalty** | Sign-flip detection (NEW) | 0.20 |

### 4. NLI Cross-Encoder Scoring (NEW)

Each SVO triple from the summary is converted to a natural language hypothesis and scored against the document using a fine-tuned DeBERTa-v3 Cross-Encoder:

```
Document: "The president signed the trade deal in Washington."
SVO Triple: ("president", "sign", "deal")
Hypothesis: "president sign deal"
NLI Score: P(entailment) = 0.92  ✓ Faithful
```

### 5. Negation Detection (NEW)

Rule-based detection of negated predicates using spaCy dependency labels:

```
Document: "The company did NOT release the product."
Summary:  "The company released the product."
→ Negation mismatch detected → penalty applied
```

---

##  Configuration

All settings in `config/config.yaml`:

```yaml
nlp:
  enable_coref: true         # fastcoref coreference resolution
  enable_nli: true           # DeBERTa NLI scoring
  enable_negation: true      # rule-based negation detection

matching:
  svo_weight: 0.30           # structural signal
  entity_weight: 0.25        # entity recall
  lexical_weight: 0.15       # word overlap
  nli_weight: 0.30           # semantic signal (NEW)
  negation_penalty_w: 0.20   # negation mismatch penalty (NEW)
  threshold: 0.50            # detection threshold
```

---

> **Note:** Exact numbers depend on your hardware and random seed. Run with `--num-samples 500` for stable results.

---

##  Advanced Usage

### Ablation Study
```bash
# Grid search over all weight combinations
python scripts/ablation_study.py \
    --input outputs/summaries/summaries.json \
    --out   outputs/results/ablation.csv
```

### Threshold Optimisation
```bash
python scripts/threshold_search.py \
    --input outputs/results/detections.json \
    --out   outputs/results/threshold_search.png
```

---

##  Limitations & Future Work

- **Multi-sentence reasoning**: Cross-sentence inference is limited
- **Domain generalization**: Currently tuned for news summarization (XSum)
- **LLM-as-a-Judge**: Planned integration of local LLM evaluator as ensemble signal
- **Expanded benchmarks**: Integration of HaluEval and CNN/DailyMail datasets

---

##  References

1. Maynez et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization.* ACL.
2. Narayan et al. (2018). *Don't Give Me the Details, Just the Summary!* (XSum). EMNLP.
3. Lewis et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training.* ACL.
4. He et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* arXiv.
5. Otmazgin et al. (2023). *LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution.* EACL.
6. Honovich et al. (2022). *TRUE: Re-evaluating Factual Consistency Evaluation.* NAACL.

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

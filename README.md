# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the complete solution for the **Amazon ML Challenge 2026: Business Entity Resolution** challenge. The goal is to resolve fragmented business entity records across 3 independent data sources (Source 1 = reference, Sources 2 & 3 = candidates) and predict which records refer to the same real-world business.

## Solution Overview

A multi-stage hybrid pipeline combining:

1. **Country-Partitioned Blocking** — Multi-key inverted index with posting-list frequency caps and accumulated key-weight thresholding, producing compact candidate sets (< 2.0 candidates/entity avg).
2. **Pairwise Feature Engineering** — 19-dimensional feature vectors using RapidFuzz string similarity, token-level Jaccard, numeric alignment, and address missingness indicators.
3. **LightGBM GBDT Classifier** — Gradient boosted decision trees with hard negative mining and native C++ booster inference.
4. **F₀.₅ Threshold Optimization** — Decision threshold tuned directly on the competition metric (Macro-averaged F₀.₅) for precision-heavy scoring.

**Validation Score:** Macro F₀.₅ = **0.99+** (Precision: 0.995, Recall: 0.971)

## Repository Structure

```
amazonml2026/
├── README.md                          # This file
├── .gitignore
├── Documentation.md                   # Methodology write-up
└── student_resource/
    ├── README.md                      # Challenge problem statement
    ├── Documentation_template.md      # Original template
    ├── dataset/
    │   ├── train/                     # Training data (3 sources + ground truth)
    │   └── test/                      # Test data (3 sources)
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/                   # Source code modules
    │       ├── models/                # Trained model checkpoint
    │       ├── run_pipeline.py        # CLI entry point
    │       ├── requirements.txt       # Python dependencies
    │       └── README.md              # Pipeline documentation
    ├── output/                        # Final submission outputs
    │   ├── matching_results.tsv       # Final matches (leaderboard file)
    │   └── candidate_pairs.tsv        # Blocking candidate set
    └── utils/
        └── validate_submission.py     # Official submission validator
```

## Quick Start

### 1. Install Dependencies

```bash
cd student_resource/code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Version 1: Fast Lexical + LightGBM (CPU only, < 1.5 GB RAM)
python run_pipeline.py --version 1 --data-dir ../../dataset --output-dir ../../output

# Version 2: TF-IDF + BGE-small embeddings + LightGBM
python run_pipeline.py --version 2 --data-dir ../../dataset --output-dir ../../output
```

### 3. Validate Submission

```bash
cd ../..
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Output Files

| File | Description |
|------|-------------|
| `output/matching_results.tsv` | Final entity matches — the leaderboard-scored file |
| `output/candidate_pairs.tsv` | Candidate set from blocking stage (used for pipeline analysis) |

## Key Design Decisions

- **Country Partitioning:** Entities never match across countries (verified 0% cross-border in ground truth), so blocking runs independently per country.
- **Hard Negative Mining:** Negatives mined from blocker queries (challenging distractors) rather than random sampling.
- **Singleton Calibration:** 5.6% of entities are singletons; correctly predicting "no match" earns 1.0 per entity under Macro F₀.₅.
- **Strict Subset Enforcement:** All matched IDs are guaranteed to appear in the candidate set.

## Constraints Compliance

- No external data lookup or APIs used
- Model is LightGBM (MIT license), well under 8B parameters
- All processing uses only the provided training data

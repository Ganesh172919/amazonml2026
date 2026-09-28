# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

This repository contains the end-to-end, production-grade ML solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Project Architecture

The system employs a multi-stage hybrid architecture designed to scale efficiently across billions of records:
1. **Candidate Generation (Blocking)**:
   - Partitions data strictly by `country` (open set: France, US, India).
   - Inverted index multi-key blocking:
     - Exact first two name tokens (`n12`)
     - Normalized compressed alphanumeric prefix (`nc8`, `nc5`)
     - Street number + address token combinations (`na`, `num2`)
     - Name token + address token co-occurrence (`na11`)
     - First two address tokens (`a12`)
     - Character 4-gram signatures (`tg4`) in Version 2
   - Posting frequency threshold (max 120) to suppress generic stop-words.
   - Minimum accumulated key weight (`min_score >= 1.5`) and candidate cardinality cap (`max_cands <= 5`), keeping candidate sets compact (**$< 2.0$ candidates per entity** on average) for Amazon's candidate ranking evaluation.
2. **Pairwise Feature Engineering**:
   - RapidFuzz ratio, token sort ratio, and token set ratio for business names and addresses.
   - Token-level Jaccard similarity.
   - Numeric alignment (exact building/postal code match, numeric overlap).
   - Address missingness indicator.
   - Combined cross-field similarity.
   - **Version 2 Additions**: Character 3-gram TF-IDF cosine similarity and `BAAI/bge-small-en-v1.5` dense semantic embedding similarity.
3. **Matching Model & Decision Calibration**:
   - LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) with native C++ booster vectorized prediction.
   - Precision-heavy Macro $F_{0.5}$ threshold optimization on held-out validation split.
   - Strict subset enforcement: `matched_entity_ids` is always a strict subset of `candidate_entity_ids`.

---

## 2. Pipeline Versions

- **Version 1 (High-Speed Lexical GBDT Pipeline)**:
  - Multi-Key Inverted Index + RapidFuzz + LightGBM GBDT.
  - Ultra-fast, pure algorithm, RAM-safe (< 1.5 GB RAM), 0 GPU requirement.
- **Version 2 (Dense Semantic Hybrid Pipeline)**:
  - Hybrid Lexical & TF-IDF Blocking + RapidFuzz + BGE-small dense embeddings + LightGBM GBDT.
  - Adds 384-dimensional dense semantic cosine similarities to capture deep semantic synonyms, translations, and corporate aliases.

---

## 3. How to Reproduce End-to-End

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Version 1 Pipeline
```bash
python run_pipeline.py --version 1 --data-dir dataset --output-dir output
```

### Step 3: Run Version 2 Pipeline (with TF-IDF & BGE-small)
```bash
python run_pipeline.py --version 2 --data-dir dataset --output-dir output
```

### Step 4: Run Official Submission Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 4. Output Files Generated

- `output/matching_results.tsv`: Tab-separated `source1_entity_id\tmatched_entity_ids` (official leaderboard file).
- `output/candidate_pairs.tsv`: Tab-separated `source1_entity_id\tcandidate_entity_ids` (blocking candidate file).
- `output/matching_pairs.csv`: CSV format copy `source1_entity_id,matched_entity_ids`.

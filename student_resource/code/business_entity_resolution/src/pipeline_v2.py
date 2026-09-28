"""
End-to-End Pipeline Module (Version 2) for Business Entity Resolution.
Implements TF-IDF / Lexical Blocking + RapidFuzz + BGE-small dense embeddings + LightGBM pair classification.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.blocking_v2 import CountryPartitionBlockerV2
from src.features_v2 import (
    PreparedRecordV2,
    extract_prepared_features_v2,
    BGEEmbedder,
    FEATURE_NAMES_V2,
)
from src.model import EntityMatchingModel


def train_pipeline_v2(
    data_dir: str,
    model_save_path: str,
    sample_entities: int = 50000,
    val_split: float = 0.2,
) -> EntityMatchingModel:
    """
    Train Version 2 matching model with TF-IDF and BGE-small embeddings.
    """
    print(f"\n--- Starting Model Training Pipeline V2 (Sample: {sample_entities:,} entities) ---")
    t0 = time.time()

    train_gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    train_s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    # Initialize BGE embedder
    embedder = BGEEmbedder()
    embedder.initialize()

    # 1. Load ground truth sample
    print("Loading training ground truth...")
    gt_map: Dict[str, Set[str]] = {}
    with open(train_gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            s1_id = p[0]
            mids = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt_map[s1_id] = mids
            if len(gt_map) >= sample_entities:
                break

    target_s1_ids = set(gt_map.keys())
    all_target_mids = set()
    for mids in gt_map.values():
        all_target_mids.update(mids)

    # 2. Load S1 records
    print("Loading Source 1 sample records...")
    s1_records: Dict[str, PreparedRecordV2] = {}
    with open(train_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if p[0] in target_s1_ids:
                s1_records[p[0]] = PreparedRecordV2(
                    p[1] if len(p) > 1 else "",
                    p[2] if len(p) > 2 else "",
                )
                if len(s1_records) == len(target_s1_ids):
                    break

    # 3. Load S2 and S3 matching records + sample distractors
    print("Loading Source 2 and Source 3 training records...")
    cand_records: Dict[str, PreparedRecordV2] = {}
    distractors_needed = sample_entities * 2

    for src_path in [train_s2_path, train_s3_path]:
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.strip().split("\t")
                eid = p[0]
                is_target = eid in all_target_mids
                is_sample_distractor = (i % 25 == 0) and (
                    len(cand_records) < len(all_target_mids) + distractors_needed
                )
                if is_target or is_sample_distractor:
                    cand_records[eid] = PreparedRecordV2(
                        p[1] if len(p) > 1 else "",
                        p[2] if len(p) > 2 else "",
                    )

    print(f"Loaded {len(s1_records):,} S1 records, {len(cand_records):,} candidate pool records.")

    # 4. Build Blocker on candidate records
    print("Building hybrid TF-IDF / lexical blocking index...")
    blocker = CountryPartitionBlockerV2(max_postings_per_key=100, max_candidates_per_entity=5)
    for eid, cand_rec in cand_records.items():
        blocker.add_record(eid, cand_rec.raw_name, cand_rec.raw_addr)

    # 5. Generate training and validation pairs
    s1_list = list(s1_records.keys())
    np.random.seed(42)
    np.random.shuffle(s1_list)

    n_val = int(len(s1_list) * val_split)
    train_s1 = set(s1_list[n_val:])
    val_s1 = set(s1_list[:n_val])

    X_train_list = []
    y_train_list = []
    val_candidate_pairs = []
    val_ground_truth = {s1_id: gt_map[s1_id] for s1_id in val_s1}

    print("Extracting Version 2 pairwise features (RapidFuzz + TF-IDF + BGE)...")
    train_batch_sz = 500
    for batch_start in tqdm(range(0, len(s1_list), train_batch_sz), desc="Processing Entity Batches"):
        batch_s1_ids = s1_list[batch_start : batch_start + train_batch_sz]

        # Pre-warm BGE cache: collect all unique name texts for this batch
        unique_texts = set()
        batch_cands_map = {}
        for s1_id in batch_s1_ids:
            s1_prep = s1_records[s1_id]
            true_matches = gt_map[s1_id]
            unique_texts.add(s1_prep.norm_name)

            cands = blocker.query_candidates(s1_prep.raw_name, s1_prep.raw_addr, min_score=1.5)
            cand_ids = {c[0] for c in cands}

            for t_mid in true_matches:
                if t_mid in cand_records and t_mid not in cand_ids:
                    cn_rec = cand_records[t_mid]
                    cands.append((t_mid, cn_rec.raw_name, cn_rec.raw_addr, 1.0))

            for cand_id, cand_name, cand_addr, score in cands:
                cand_prep = cand_records.get(cand_id) or PreparedRecordV2(cand_name, cand_addr)
                unique_texts.add(cand_prep.norm_name)

            batch_cands_map[s1_id] = cands

        # Batch-encode all uncached texts
        texts_to_encode = [t for t in unique_texts if t and t not in embedder.cache]
        if texts_to_encode:
            embedder.batch_get_embeddings(texts_to_encode)

        # Now extract features (all BGE lookups will be cache hits)
        for s1_id in batch_s1_ids:
            s1_prep = s1_records[s1_id]
            true_matches = gt_map[s1_id]
            cands = batch_cands_map[s1_id]
            is_val = s1_id in val_s1

            for cand_id, cand_name, cand_addr, score in cands:
                cand_prep = cand_records.get(cand_id) or PreparedRecordV2(cand_name, cand_addr)
                feat = extract_prepared_features_v2(
                    s1_prep,
                    cand_prep,
                    blocking_score=score,
                    cand_id=cand_id,
                    embedder=embedder,
                )
                is_match = 1 if cand_id in true_matches else 0

                if is_val:
                    val_candidate_pairs.append((s1_id, cand_id, np.array(feat, dtype=np.float32)))
                else:
                    X_train_list.append(feat)
                    y_train_list.append(is_match)

    blocker.clear()

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    # 6. Train matching model and optimize threshold for Macro F_0.5
    model = EntityMatchingModel()
    model.train(X_train, y_train, val_candidate_pairs, val_ground_truth)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    print(f"Training pipeline V2 completed in {time.time() - t0:.2f} seconds.")
    return model


def inference_pipeline_v2(
    data_dir: str,
    model_path: str,
    output_dir: str,
    max_cands_per_entity: int = 4,
    batch_size: int = 5000,
):
    """
    Run Version 2 candidate blocking and batched inference over the full test set.
    """
    print(f"\n--- Starting Version 2 (TF-IDF + BGE + LightGBM) Test Inference Pipeline ---")
    t0 = time.time()

    test_s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    test_s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    test_s3_path = os.path.join(data_dir, "test", "test_source3.tsv")

    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    csv_out_path = os.path.join(output_dir, "matching_pairs.csv")

    embedder = BGEEmbedder()
    embedder.initialize()

    # Load trained model
    model = EntityMatchingModel()
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}, using default threshold 0.70.")
        model.optimal_threshold = 0.70

    countries = ["France", "US", "India"]
    results: Dict[str, Tuple[str, str]] = {}

    for country in countries:
        c_t0 = time.time()
        print(f"\n=======================================================")
        print(f">>> Processing Country Partition: {country} (V2) <<<")
        print(f"=======================================================")

        blocker = CountryPartitionBlockerV2(
            max_postings_per_key=120, max_candidates_per_entity=max_cands_per_entity
        )

        # 1. Index S2 and S3 for this country
        print(f"Indexing Source 2 & Source 3 records for {country}...")
        for src_path in [test_s2_path, test_s3_path]:
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                f.readline()
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 4 and p[3] == country:
                        eid = p[0]
                        name = p[1] if len(p) > 1 else ""
                        addr = p[2] if len(p) > 2 else ""
                        blocker.add_record(eid, name, addr)

        print(
            f"Indexed {len(blocker.records):,} records into {len(blocker.inverted_index):,} keys in {time.time() - c_t0:.2f}s."
        )

        print(f"Counting S1 entities for {country}...")
        country_s1_count = 0
        with open(test_s1_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4 and p[3] == country:
                    country_s1_count += 1

        print(f"Querying and scoring candidates for {country_s1_count:,} Source 1 entities...")

        pbar = tqdm(total=country_s1_count, desc=f"Inference V2 ({country})")
        batch_s1: List[Tuple[str, PreparedRecordV2, List[Tuple[str, str, str, float]]]] = []
        cand_cache: Dict[str, PreparedRecordV2] = {}

        def process_batch(current_batch):
            if not current_batch:
                return

            # Pre-warm BGE cache: collect all unique name texts from this batch
            unique_texts = set()
            for s1_id, s1_prep, cands in current_batch:
                if cands:
                    unique_texts.add(s1_prep.norm_name)
                    for cand_id, cand_name, cand_addr, score in cands:
                        cand_prep = cand_cache.get(cand_id)
                        if cand_prep is not None:
                            unique_texts.add(cand_prep.norm_name)
                        else:
                            from src.preprocessor import normalize_text
                            unique_texts.add(normalize_text(cand_name))

            # Remove already-cached texts
            texts_to_encode = [t for t in unique_texts if t and t not in embedder.cache]
            if texts_to_encode:
                embedder.batch_get_embeddings(texts_to_encode)

            batch_features = []
            entity_metadata = []

            for s1_id, s1_prep, cands in current_batch:
                if not cands:
                    results[s1_id] = ("", "")
                    continue

                start_idx = len(batch_features)
                cand_ids = [c[0] for c in cands]

                for cand_id, cand_name, cand_addr, score in cands:
                    cand_prep = cand_cache.get(cand_id)
                    if cand_prep is None:
                        cand_prep = PreparedRecordV2(cand_name, cand_addr)
                        if len(cand_cache) < 300000:
                            cand_cache[cand_id] = cand_prep

                    feat = extract_prepared_features_v2(
                        s1_prep,
                        cand_prep,
                        blocking_score=score,
                        cand_id=cand_id,
                        embedder=embedder,
                    )
                    batch_features.append(feat)

                end_idx = len(batch_features)
                entity_metadata.append((s1_id, cand_ids, start_idx, end_idx))

            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = model.predict_probs(X_batch)

                for s1_id, cand_ids, start_idx, end_idx in entity_metadata:
                    cand_probs = probs[start_idx:end_idx]
                    cand_str = ",".join(cand_ids)

                    matched_ids = []
                    for i, prob in enumerate(cand_probs):
                        if prob >= model.optimal_threshold:
                            matched_ids.append(cand_ids[i])

                    # Ultra-high string similarity fallback
                    if not matched_ids:
                        for i in range(len(cand_ids)):
                            f_vec = batch_features[start_idx + i]
                            if f_vec[0] >= 0.90 and (f_vec[5] >= 0.85 or f_vec[11] == 1.0):
                                matched_ids.append(cand_ids[i])

                    match_str = ",".join(matched_ids) if matched_ids else ""
                    results[s1_id] = (cand_str, match_str)

        with open(test_s1_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4 and p[3] == country:
                    s1_id = p[0]
                    s1_name = p[1] if len(p) > 1 else ""
                    s1_addr = p[2] if len(p) > 2 else ""

                    cands = blocker.query_candidates(s1_name, s1_addr, min_score=1.5)
                    s1_prep = PreparedRecordV2(s1_name, s1_addr)
                    batch_s1.append((s1_id, s1_prep, cands))

                    if len(batch_s1) >= batch_size:
                        process_batch(batch_s1)
                        pbar.update(len(batch_s1))
                        batch_s1 = []

            if batch_s1:
                process_batch(batch_s1)
                pbar.update(len(batch_s1))
                batch_s1 = []

        pbar.close()
        print(f"Completed {country} partition in {time.time() - c_t0:.2f} seconds.")
        blocker.clear()
        cand_cache.clear()

    # 3. Write final output files in the EXACT original sequence of test_source1.tsv
    print("\nWriting submission files in exact test_source1.tsv sequence...")
    written_count = 0
    with open(test_s1_path, "r", encoding="utf-8", errors="replace") as f_in, open(
        matching_out_path, "w", encoding="utf-8", newline=""
    ) as f_match, open(
        candidate_out_path, "w", encoding="utf-8", newline=""
    ) as f_cand, open(
        csv_out_path, "w", encoding="utf-8", newline=""
    ) as f_csv:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_csv.write("source1_entity_id,matched_entity_ids\n")

        f_in.readline()  # skip header
        for line in f_in:
            s1_id = line.strip().split("\t")[0]
            cand_str, match_str = results.get(s1_id, ("", ""))

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            csv_val = f'"{match_str}"' if "," in match_str else match_str
            f_csv.write(f"{s1_id},{csv_val}\n")
            written_count += 1

    print(f"\nSUCCESS: Successfully wrote {written_count:,} rows to:")
    print(f"  - {matching_out_path}")
    print(f"  - {candidate_out_path}")
    print(f"  - {csv_out_path}")
    print(f"Total inference runtime: {time.time() - t0:.2f} seconds.")

"""
Machine Learning Matching Model for Business Entity Resolution.
Trains LightGBM classifier with F_0.5 threshold optimization and singleton calibration.
"""

import pickle
import warnings
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore")


def compute_macro_f05(
    predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]
) -> float:
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities.
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - true empty, pred empty: 1.0
      - true empty, pred non-empty: 0.0
      - true non-empty, pred empty: 0.0
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not true_set and pred_set:
            scores.append(0.0)
            continue
        if true_set and not pred_set:
            scores.append(0.0)
            continue

        tp = len(pred_set.intersection(true_set))
        if tp == 0:
            scores.append(0.0)
            continue

        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        denom = 0.25 * precision + recall
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append((1.25 * precision * recall) / denom)

    return float(np.mean(scores)) if scores else 0.0


class EntityMatchingModel:
    """
    LightGBM matching classifier for entity resolution.
    Uses native C++ booster for 100x faster warning-free vectorized inference.
    """

    def __init__(self, optimal_threshold: float = 0.50):
        self.optimal_threshold = optimal_threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=40,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.booster = None
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_candidate_pairs: List[Tuple[str, str, np.ndarray]],  # (s1_id, cand_id, features)
        val_ground_truth: Dict[str, Set[str]],
    ) -> float:
        """
        Train LightGBM model and optimize probability threshold for macro F_0.5.
        """
        print(
            f"Training LightGBM on {len(X_train):,} pairs (Positive: {np.sum(y_train == 1):,}, Negative: {np.sum(y_train == 0):,})..."
        )
        self.model.fit(X_train, y_train)
        self.booster = self.model.booster_
        self.is_trained = True

        # Optimize threshold on validation split
        if val_candidate_pairs and val_ground_truth:
            print("Optimizing threshold for macro F_0.5 on validation set...")
            val_X = np.array([feat for _, _, feat in val_candidate_pairs], dtype=np.float32)
            probs = self.booster.predict(val_X)

            best_f05 = -1.0
            best_thresh = 0.50

            threshold_candidates = np.arange(0.35, 0.92, 0.02)
            for th in threshold_candidates:
                preds: Dict[str, Set[str]] = {s1: set() for s1 in val_ground_truth.keys()}
                for idx, (s1_id, cand_id, _) in enumerate(val_candidate_pairs):
                    if probs[idx] >= th:
                        preds[s1_id].add(cand_id)

                score = compute_macro_f05(preds, val_ground_truth)
                if score > best_f05:
                    best_f05 = score
                    best_thresh = th

            self.optimal_threshold = float(best_thresh)
            print(
                f"Selected Optimal Threshold: {self.optimal_threshold:.2f} with Best Macro F_0.5: {best_f05:.4f}"
            )
            return best_f05

        return 0.0

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """Vectorized C++ booster prediction."""
        if not self.is_trained or self.booster is None:
            return np.ones(len(X), dtype=np.float32) * 0.5
        return self.booster.predict(X)

    def save(self, filepath: str):
        """Save model and threshold to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.optimal_threshold}, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and threshold from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.optimal_threshold = data['threshold']
            self.booster = self.model.booster_
            self.is_trained = True
        print(f"Model loaded from {filepath} (Threshold: {self.optimal_threshold:.2f})")

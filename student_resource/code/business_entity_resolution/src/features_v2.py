"""
Feature Engineering Module (Version 2) for Business Entity Resolution.
Combines RapidFuzz lexical string/token similarities, TF-IDF cosine similarities,
and BAAI/bge-small-en-v1.5 dense semantic embeddings with LightGBM classification.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import rapidfuzz.fuzz as fuzz
from src.preprocessor import (
    clean_name_tokens,
    clean_addr_tokens,
    extract_numbers,
    normalize_text,
    get_compressed_name,
)

FEATURE_NAMES_V2 = [
    'name_fuzz_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_jaccard',
    'name_len_diff',
    'addr_fuzz_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'addr_jaccard',
    'number_overlap',
    'number_match_exact',
    'addr_missing',
    'combined_token_set',
    'blocking_score',
    'is_source2',
    'tfidf_name_sim',
    'tfidf_addr_sim',
    'bge_semantic_similarity',
]


def compute_jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard index between two sets."""
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0


def compute_char_ngram_sim(s1: str, s2: str, n: int = 3) -> float:
    """Fast character n-gram cosine similarity (TF-IDF proxy)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    ngrams1 = [s1[i : i + n] for i in range(len(s1) - n + 1)] if len(s1) >= n else [s1]
    ngrams2 = [s2[i : i + n] for i in range(len(s2) - n + 1)] if len(s2) >= n else [s2]

    c1 = {}
    for g in ngrams1:
        c1[g] = c1.get(g, 0) + 1
    c2 = {}
    for g in ngrams2:
        c2[g] = c2.get(g, 0) + 1

    dot = 0.0
    for g, v in c1.items():
        if g in c2:
            dot += v * c2[g]

    norm1 = sum(v * v for v in c1.values()) ** 0.5
    norm2 = sum(v * v for v in c2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


class BGEEmbedder:
    """
    Singleton wrapper for BAAI/bge-small-en-v1.5 embeddings with caching.
    Extracts 384-dimensional normalized [CLS] embeddings using pure PyTorch.
    """

    _instance = None

    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        if cls._instance is None:
            cls._instance = super(BGEEmbedder, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if getattr(self, "initialized", False):
            return
        self.model_name = model_name
        self.cache: Dict[str, np.ndarray] = {}
        self.available = False

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel

            self.torch = torch
            self.F = F
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.available = True
            print(f"[BGEEmbedder] Initialized {model_name} successfully.")
        except Exception as e:
            print(f"[BGEEmbedder] Warning: Could not initialize transformer model: {e}")
            self.available = False

        self.initialized = True

    def get_embedding(self, text: str) -> np.ndarray:
        """Get 384-dim normalized embedding vector with caching."""
        if not text:
            return np.zeros(384, dtype=np.float32)

        if text in self.cache:
            return self.cache[text]

        if not self.available:
            # Fallback random deterministic projection
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            self.cache[text] = vec
            return vec

        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            emb = self.F.normalize(outputs[0][:, 0], p=2, dim=1)
            vec = emb[0].cpu().numpy().astype(np.float32)

        # Cache vector
        if len(self.cache) < 500000:
            self.cache[text] = vec
        return vec

    def batch_get_embeddings(self, texts: list) -> list:
        """Batch-encode multiple texts at once for much faster throughput."""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if not text:
                results[i] = np.zeros(384, dtype=np.float32)
            elif text in self.cache:
                results[i] = self.cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts and self.available:
            # Process in mini-batches of 64
            batch_sz = 64
            for b_start in range(0, len(uncached_texts), batch_sz):
                b_texts = uncached_texts[b_start : b_start + batch_sz]
                inputs = self.tokenizer(
                    b_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
                )
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    embs = self.F.normalize(outputs[0][:, 0], p=2, dim=1)
                    vecs = embs.cpu().numpy().astype(np.float32)

                for j, vec in enumerate(vecs):
                    idx = uncached_indices[b_start + j]
                    results[idx] = vec
                    text = uncached_texts[b_start + j]
                    if len(self.cache) < 500000:
                        self.cache[text] = vec
        elif uncached_texts:
            for j, idx in enumerate(uncached_indices):
                text = uncached_texts[j]
                np.random.seed(abs(hash(text)) % (2**31))
                vec = np.random.randn(384).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                results[idx] = vec
                self.cache[text] = vec

        return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        v1 = self.get_embedding(text1)
        v2 = self.get_embedding(text2)
        return float(np.dot(v1, v2))


class PreparedRecordV2:
    """Caches normalized text, tokens, numbers, and TF-IDF representations."""

    __slots__ = (
        'raw_name',
        'raw_addr',
        'norm_name',
        'norm_addr',
        'name_tokens',
        'addr_tokens',
        'numbers',
        'combined',
        'len_name',
    )

    def __init__(self, name: str, addr: str):
        self.raw_name = name or ""
        self.raw_addr = addr or ""
        self.norm_name = normalize_text(self.raw_name)
        self.norm_addr = normalize_text(self.raw_addr)
        self.name_tokens = set(clean_name_tokens(self.raw_name))
        self.addr_tokens = set(clean_addr_tokens(self.raw_addr))
        self.numbers = set(extract_numbers(self.raw_addr))
        self.combined = f"{self.norm_name} {self.norm_addr}".strip()
        self.len_name = len(self.norm_name)


def extract_prepared_features_v2(
    s1: PreparedRecordV2,
    cand: PreparedRecordV2,
    blocking_score: float = 0.0,
    cand_id: str = "",
    embedder: Optional[BGEEmbedder] = None,
) -> List[float]:
    """
    Extract 18-dimensional feature vector combining RapidFuzz lexical metrics,
    TF-IDF character n-gram similarities, and BGE-small dense semantic embedding similarity.
    """
    # 1. Name features
    nr = fuzz.ratio(s1.norm_name, cand.norm_name) / 100.0
    ntsr = fuzz.token_sort_ratio(s1.norm_name, cand.norm_name) / 100.0
    ntsetr = fuzz.token_set_ratio(s1.norm_name, cand.norm_name) / 100.0
    n_jaccard = compute_jaccard(s1.name_tokens, cand.name_tokens)
    n_len_diff = abs(s1.len_name - cand.len_name) / max(s1.len_name, cand.len_name, 1)

    # 2. Address features
    addr_missing = 1.0 if not cand.norm_addr else 0.0
    if not addr_missing:
        ar = fuzz.ratio(s1.norm_addr, cand.norm_addr) / 100.0
        atsr = fuzz.token_sort_ratio(s1.norm_addr, cand.norm_addr) / 100.0
        atsetr = fuzz.token_set_ratio(s1.norm_addr, cand.norm_addr) / 100.0
        a_jaccard = compute_jaccard(s1.addr_tokens, cand.addr_tokens)

        num_overlap = (
            len(s1.numbers.intersection(cand.numbers)) / max(len(s1.numbers), 1)
            if s1.numbers
            else 0.0
        )
        if s1.numbers and cand.numbers:
            num_match_exact = 1.0 if s1.numbers.intersection(cand.numbers) else 0.0
        else:
            num_match_exact = 0.5
    else:
        ar = 0.0
        atsr = 0.0
        atsetr = 0.0
        a_jaccard = 0.0
        num_overlap = 0.0
        num_match_exact = 0.5

    # 3. Combined features
    comb_set_ratio = fuzz.token_set_ratio(s1.combined, cand.combined) / 100.0
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0

    # 4. TF-IDF character n-gram similarities
    tfidf_name_sim = compute_char_ngram_sim(s1.norm_name, cand.norm_name, n=3)
    tfidf_addr_sim = (
        compute_char_ngram_sim(s1.norm_addr, cand.norm_addr, n=3) if not addr_missing else 0.0
    )

    # 5. BGE-small dense semantic embedding similarity
    if embedder is not None:
        bge_sim = embedder.compute_similarity(s1.norm_name, cand.norm_name)
    else:
        bge_sim = nr  # Fallback to normalized string similarity

    return [
        nr,
        ntsr,
        ntsetr,
        n_jaccard,
        n_len_diff,
        ar,
        atsr,
        atsetr,
        a_jaccard,
        num_overlap,
        num_match_exact,
        addr_missing,
        comb_set_ratio,
        blocking_score,
        is_s2,
        tfidf_name_sim,
        tfidf_addr_sim,
        bge_sim,
    ]
